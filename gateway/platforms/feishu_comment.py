"""
Feishu/Lark drive document comment handling.

Processes ``drive.notice.comment_add_v1`` events and interacts with the
Drive v2 comment reaction API.  Kept in a separate module so that the
main ``feishu.py`` adapter does not grow further and comment-related
logic can evolve independently.

Flow:
  1. Parse event -> extract file_token, comment_id, reply_id, etc.
  2. Add OK reaction
  3. Parallel fetch: doc meta + comment details (batch_query)
  4. Branch on is_whole:
       Whole -> list whole comments timeline
       Local -> list comment thread replies
  5. Build prompt (local or whole)
  6. Run AIAgent with no feishu tools.  If the agent decides it needs
     document text to reply, it emits a ``<NEED_DOC_READ>{...}`` sentinel;
     business code then fetches the requested docs (from a whitelist of
     the source doc + comment-referenced docs) and re-invokes the agent
     with the content appended.  See ``_run_comment_agent``.
  7. Route reply:
       Whole -> add_whole_comment
       Local -> reply_to_comment (fallback to add_whole_comment on 1069302)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# External dependencies bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommentContext:
    """External dependencies the comment handler needs from its adapter.

    Boundary dataclass: shields the handler from the concrete
    ``FeishuAdapter`` type and its private-attribute layout.  Everything
    the handler reads off the adapter flows through this struct, which
    keeps the handler signature stable across adapter refactors and makes
    tests cheaper (construct a 3-field dataclass instead of mocking an
    adapter).
    """
    client: Any                     # lark_oapi client (for Feishu API calls)
    session_store: Optional[Any]    # gateway SessionStore; may be None in
                                    # degraded runtimes or stateless tests
    self_open_id: str               # bot's own open_id — used to filter
                                    # self-authored events and to strip
                                    # routing @mentions from timeline text

    @classmethod
    def from_adapter(cls, adapter: Any, *, self_open_id: str = "") -> "CommentContext":
        """Build a ``CommentContext`` from a live ``FeishuAdapter``.

        Concentrates all ``adapter._xxx`` reads into this one method — if
        the adapter later grows public getters, only this factory changes.
        """
        return cls(
            client=adapter._client,
            session_store=getattr(adapter, "_session_store", None),
            self_open_id=self_open_id,
        )


# ---------------------------------------------------------------------------
# Lark SDK helpers (lazy-imported)
# ---------------------------------------------------------------------------


def _build_request(method: str, uri: str, paths=None, queries=None, body=None):
    """Build a lark_oapi BaseRequest."""
    from lark_oapi import AccessTokenType
    from lark_oapi.core.enum import HttpMethod
    from lark_oapi.core.model.base_request import BaseRequest

    http_method = HttpMethod.GET if method == "GET" else HttpMethod.POST

    builder = (
        BaseRequest.builder()
        .http_method(http_method)
        .uri(uri)
        .token_types({AccessTokenType.TENANT})
    )
    if paths:
        builder = builder.paths(paths)
    if queries:
        builder = builder.queries(queries)
    if body is not None:
        builder = builder.body(body)
    return builder.build()


async def _exec_request(client, method, uri, paths=None, queries=None, body=None):
    """Execute a lark API request and return (code, msg, data_dict)."""
    # Log metadata only — request bodies may contain user content (reply text,
    # comment text) which must not land in persistent logs.
    body_bytes = len(json.dumps(body, ensure_ascii=False).encode("utf-8")) if body else 0
    logger.info("[Feishu-Comment] API >>> %s %s paths=%s queries=%s body_bytes=%d",
                 method, uri, paths, queries, body_bytes)
    request = _build_request(method, uri, paths, queries, body)
    response = await asyncio.to_thread(client.request, request)

    code = getattr(response, "code", None)
    msg = getattr(response, "msg", "")

    data: dict = {}
    raw = getattr(response, "raw", None)
    if raw and hasattr(raw, "content"):
        try:
            body_json = json.loads(raw.content)
            data = body_json.get("data", {})
        except (json.JSONDecodeError, AttributeError):
            pass
    if not data:
        resp_data = getattr(response, "data", None)
        if isinstance(resp_data, dict):
            data = resp_data
        elif resp_data and hasattr(resp_data, "__dict__"):
            data = vars(resp_data)

    logger.info("[Feishu-Comment] API <<< %s %s code=%s msg=%s data_keys=%s",
                 method, uri, code, msg, list(data.keys()) if data else "empty")
    if code != 0:
        # Raw response bodies may echo user content back in error messages;
        # log only the code + msg we've already extracted above.
        logger.warning("[Feishu-Comment] API FAIL: %s %s code=%s msg=%s",
                       method, uri, code, msg)
    return code, msg, data


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------


def parse_drive_comment_event(data: Any) -> Optional[Dict[str, Any]]:
    """Extract structured fields from a ``drive.notice.comment_add_v1`` payload.

    *data* may be a ``CustomizedEvent`` (WebSocket) whose ``.event`` is a dict,
    or a ``SimpleNamespace`` (Webhook) built from the full JSON body.

    Returns a flat dict with the relevant fields, or ``None`` when the
    payload is malformed.
    """
    logger.debug("[Feishu-Comment] parse_drive_comment_event: data type=%s", type(data).__name__)
    event = getattr(data, "event", None)
    if event is None:
        logger.debug("[Feishu-Comment] parse_drive_comment_event: no .event attribute, returning None")
        return None

    evt: dict = event if isinstance(event, dict) else (
        vars(event) if hasattr(event, "__dict__") else {}
    )
    logger.debug("[Feishu-Comment] parse_drive_comment_event: evt keys=%s", list(evt.keys()))

    notice_meta = evt.get("notice_meta") or {}
    if not isinstance(notice_meta, dict):
        notice_meta = vars(notice_meta) if hasattr(notice_meta, "__dict__") else {}

    from_user = notice_meta.get("from_user_id") or {}
    if not isinstance(from_user, dict):
        from_user = vars(from_user) if hasattr(from_user, "__dict__") else {}

    to_user = notice_meta.get("to_user_id") or {}
    if not isinstance(to_user, dict):
        to_user = vars(to_user) if hasattr(to_user, "__dict__") else {}

    return {
        "event_id": str(evt.get("event_id") or ""),
        "comment_id": str(evt.get("comment_id") or ""),
        "reply_id": str(evt.get("reply_id") or ""),
        "is_mentioned": bool(evt.get("is_mentioned")),
        "timestamp": str(evt.get("timestamp") or ""),
        "file_token": str(notice_meta.get("file_token") or ""),
        "file_type": str(notice_meta.get("file_type") or ""),
        "notice_type": str(notice_meta.get("notice_type") or ""),
        "from_open_id": str(from_user.get("open_id") or ""),
        "to_open_id": str(to_user.get("open_id") or ""),
    }


# ---------------------------------------------------------------------------
# Comment reaction API
# ---------------------------------------------------------------------------

_REACTION_URI = "/open-apis/drive/v2/files/:file_token/comments/reaction"


async def add_comment_reaction(
    client: Any,
    *,
    file_token: str,
    file_type: str,
    reply_id: str,
    reaction_type: str = "OK",
) -> bool:
    """Add an emoji reaction to a document comment reply.

    Uses the Drive v2 ``update_reaction`` endpoint::

        POST /open-apis/drive/v2/files/{file_token}/comments/reaction?file_type=...

    Returns ``True`` on success, ``False`` on failure (errors are logged).
    """
    try:
        from lark_oapi import AccessTokenType  # noqa: F401
    except ImportError:
        logger.error("[Feishu-Comment] lark_oapi not available")
        return False

    body = {
        "action": "add",
        "reply_id": reply_id,
        "reaction_type": reaction_type,
    }

    code, msg, _ = await _exec_request(
        client, "POST", _REACTION_URI,
        paths={"file_token": file_token},
        queries=[("file_type", file_type)],
        body=body,
    )

    succeeded = code == 0
    if succeeded:
        logger.info(
            "[Feishu-Comment] Reaction '%s' added: file=%s:%s reply=%s",
            reaction_type, file_type, file_token, reply_id,
        )
    else:
        logger.warning(
            "[Feishu-Comment] Reaction API failed: code=%s msg=%s "
            "file=%s:%s reply=%s",
            code, msg, file_type, file_token, reply_id,
        )
    return succeeded


async def delete_comment_reaction(
    client: Any,
    *,
    file_token: str,
    file_type: str,
    reply_id: str,
    reaction_type: str = "OK",
) -> bool:
    """Remove an emoji reaction from a document comment reply.

    Best-effort — errors are logged but not raised.
    """
    body = {
        "action": "delete",
        "reply_id": reply_id,
        "reaction_type": reaction_type,
    }

    code, msg, _ = await _exec_request(
        client, "POST", _REACTION_URI,
        paths={"file_token": file_token},
        queries=[("file_type", file_type)],
        body=body,
    )

    succeeded = code == 0
    if succeeded:
        logger.info(
            "[Feishu-Comment] Reaction '%s' deleted: file=%s:%s reply=%s",
            reaction_type, file_type, file_token, reply_id,
        )
    else:
        logger.warning(
            "[Feishu-Comment] Reaction API failed: code=%s msg=%s "
            "file=%s:%s reply=%s",
            code, msg, file_type, file_token, reply_id,
        )
    return succeeded


# ---------------------------------------------------------------------------
# API call layer
# ---------------------------------------------------------------------------

_BATCH_QUERY_META_URI = "/open-apis/drive/v1/metas/batch_query"
_BATCH_QUERY_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/comments/batch_query"
_LIST_COMMENTS_URI = "/open-apis/drive/v1/files/:file_token/comments"
_LIST_REPLIES_URI = "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"
_REPLY_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"
_ADD_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/new_comments"


async def query_document_meta(
    client: Any, file_token: str, file_type: str,
) -> Dict[str, Any]:
    """Fetch document title and URL via batch_query meta API.

    Returns ``{"title": "...", "url": "...", "doc_type": "..."}`` or empty dict.
    """
    body = {
        "request_docs": [{"doc_token": file_token, "doc_type": file_type}],
        "with_url": True,
    }
    logger.debug("[Feishu-Comment] query_document_meta: file_token=%s file_type=%s", file_token, file_type)
    code, msg, data = await _exec_request(
        client, "POST", _BATCH_QUERY_META_URI, body=body,
    )
    if code != 0:
        logger.warning("[Feishu-Comment] Meta batch_query failed: code=%s msg=%s", code, msg)
        return {}

    metas = data.get("metas", [])
    # Don't dump metas value — entries include title and other business info.
    logger.debug("[Feishu-Comment] query_document_meta: raw metas type=%s count=%s",
                 type(metas).__name__, len(metas) if hasattr(metas, "__len__") else "?")
    if not metas:
        # Try alternate response shape: metas may be a dict keyed by token
        if isinstance(data.get("metas"), dict):
            meta = data["metas"].get(file_token, {})
        else:
            logger.debug("[Feishu-Comment] query_document_meta: no metas found")
            return {}
    else:
        meta = metas[0] if isinstance(metas, list) else {}

    result = {
        "title": meta.get("title", ""),
        "url": meta.get("url", ""),
        "doc_type": meta.get("doc_type", file_type),
    }
    # Title may contain business-sensitive info (e.g. project names); omit.
    logger.info("[Feishu-Comment] query_document_meta: url=%s",
                result["url"][:80] if result["url"] else "")
    return result


_COMMENT_RETRY_LIMIT = 6
_COMMENT_RETRY_DELAY_S = 1.0


async def batch_query_comment(
    client: Any, file_token: str, file_type: str, comment_id: str,
) -> Dict[str, Any]:
    """Fetch comment details via batch_query comment API.

    Retries up to 6 times on failure (handles eventual consistency).

    Returns the comment dict with fields like ``is_whole``, ``quote``,
    ``reply_list``, etc.  Empty dict on failure.
    """
    logger.debug("[Feishu-Comment] batch_query_comment: file_token=%s comment_id=%s", file_token, comment_id)

    for attempt in range(_COMMENT_RETRY_LIMIT):
        code, msg, data = await _exec_request(
            client, "POST", _BATCH_QUERY_COMMENT_URI,
            paths={"file_token": file_token},
            queries=[
                ("file_type", file_type),
                ("user_id_type", "open_id"),
            ],
            body={"comment_ids": [comment_id]},
        )
        if code == 0:
            break
        if attempt < _COMMENT_RETRY_LIMIT - 1:
            logger.info(
                "[Feishu-Comment] batch_query_comment retry %d/%d: code=%s msg=%s",
                attempt + 1, _COMMENT_RETRY_LIMIT, code, msg,
            )
            await asyncio.sleep(_COMMENT_RETRY_DELAY_S)
        else:
            logger.warning(
                "[Feishu-Comment] batch_query_comment failed after %d attempts: code=%s msg=%s",
                _COMMENT_RETRY_LIMIT, code, msg,
            )
            return {}

    # Response: {"items": [{"comment_id": "...", ...}]}
    items = data.get("items", [])
    logger.debug("[Feishu-Comment] batch_query_comment: got %d items", len(items) if isinstance(items, list) else 0)
    if items and isinstance(items, list):
        item = items[0]
        # quote is user content — log length only so persistent logs don't
        # expose the quoted snippet of the document to other operators.
        quote = item.get("quote", "") or ""
        logger.info("[Feishu-Comment] batch_query_comment: is_whole=%s quote_len=%d reply_count=%s",
                    item.get("is_whole"),
                    len(quote),
                    len(item.get("reply_list", {}).get("replies", [])) if isinstance(item.get("reply_list"), dict) else "?")
        return item
    logger.warning("[Feishu-Comment] batch_query_comment: empty items, raw data keys=%s", list(data.keys()))
    return {}


async def list_whole_comments(
    client: Any, file_token: str, file_type: str,
) -> List[Dict[str, Any]]:
    """List all whole-document comments (paginated, up to 500)."""
    logger.debug("[Feishu-Comment] list_whole_comments: file_token=%s", file_token)
    all_comments: List[Dict[str, Any]] = []
    page_token = ""

    for _ in range(5):  # max 5 pages
        queries = [
            ("file_type", file_type),
            ("is_whole", "true"),
            ("page_size", "100"),
            ("user_id_type", "open_id"),
        ]
        if page_token:
            queries.append(("page_token", page_token))

        code, msg, data = await _exec_request(
            client, "GET", _LIST_COMMENTS_URI,
            paths={"file_token": file_token},
            queries=queries,
        )
        if code != 0:
            logger.warning("[Feishu-Comment] List whole comments failed: code=%s msg=%s", code, msg)
            break

        items = data.get("items", [])
        if isinstance(items, list):
            all_comments.extend(items)
            logger.debug("[Feishu-Comment] list_whole_comments: page got %d items, total=%d",
                         len(items), len(all_comments))

        if not data.get("has_more"):
            break
        page_token = data.get("page_token", "")
        if not page_token:
            break

    logger.info("[Feishu-Comment] list_whole_comments: total %d whole comments fetched", len(all_comments))
    return all_comments


async def list_comment_replies(
    client: Any, file_token: str, file_type: str, comment_id: str,
    *, expect_reply_id: str = "",
) -> List[Dict[str, Any]]:
    """List all replies in a comment thread (paginated, up to 500).

    If *expect_reply_id* is set and not found in the first fetch,
    retries up to 6 times (handles eventual consistency).
    """
    logger.debug("[Feishu-Comment] list_comment_replies: file_token=%s comment_id=%s", file_token, comment_id)

    for attempt in range(_COMMENT_RETRY_LIMIT):
        all_replies: List[Dict[str, Any]] = []
        page_token = ""
        fetch_ok = True

        for _ in range(5):  # max 5 pages
            queries = [
                ("file_type", file_type),
                ("page_size", "100"),
                ("user_id_type", "open_id"),
            ]
            if page_token:
                queries.append(("page_token", page_token))

            code, msg, data = await _exec_request(
                client, "GET", _LIST_REPLIES_URI,
                paths={"file_token": file_token, "comment_id": comment_id},
                queries=queries,
            )
            if code != 0:
                logger.warning("[Feishu-Comment] List replies failed: code=%s msg=%s", code, msg)
                fetch_ok = False
                break

            items = data.get("items", [])
            if isinstance(items, list):
                all_replies.extend(items)

            if not data.get("has_more"):
                break
            page_token = data.get("page_token", "")
            if not page_token:
                break

        # If we don't need a specific reply, or we found it, return
        if not expect_reply_id or not fetch_ok:
            break
        found = any(r.get("reply_id") == expect_reply_id for r in all_replies)
        if found:
            break
        if attempt < _COMMENT_RETRY_LIMIT - 1:
            logger.info(
                "[Feishu-Comment] list_comment_replies: reply_id=%s not found, retry %d/%d",
                expect_reply_id, attempt + 1, _COMMENT_RETRY_LIMIT,
            )
            await asyncio.sleep(_COMMENT_RETRY_DELAY_S)
        else:
            logger.warning(
                "[Feishu-Comment] list_comment_replies: reply_id=%s not found after %d attempts",
                expect_reply_id, _COMMENT_RETRY_LIMIT,
            )

    logger.info("[Feishu-Comment] list_comment_replies: total %d replies fetched", len(all_replies))
    return all_replies


def _sanitize_comment_text(text: str) -> str:
    """Escape characters not allowed in Feishu comment text_run content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def reply_to_comment(
    client: Any, file_token: str, file_type: str, comment_id: str, text: str,
) -> Tuple[bool, int]:
    """Post a reply to a local comment thread.

    Returns ``(success, code)``.
    """
    text = _sanitize_comment_text(text)
    # Reply text is the agent's generated content — log length only.
    logger.info("[Feishu-Comment] reply_to_comment: comment_id=%s text_len=%d",
                comment_id, len(text))
    body = {
        "content": {
            "elements": [
                {"type": "text_run", "text_run": {"text": text}},
            ]
        }
    }

    code, msg, _ = await _exec_request(
        client, "POST", _REPLY_COMMENT_URI,
        paths={"file_token": file_token, "comment_id": comment_id},
        queries=[("file_type", file_type)],
        body=body,
    )
    if code != 0:
        logger.warning(
            "[Feishu-Comment] reply_to_comment FAILED: code=%s msg=%s comment_id=%s",
            code, msg, comment_id,
        )
    else:
        logger.info("[Feishu-Comment] reply_to_comment OK: comment_id=%s", comment_id)
    return code == 0, code


async def add_whole_comment(
    client: Any, file_token: str, file_type: str, text: str,
) -> bool:
    """Add a new whole-document comment.

    Returns ``True`` on success.
    """
    text = _sanitize_comment_text(text)
    # Agent-generated content — log length only.
    logger.info("[Feishu-Comment] add_whole_comment: file_token=%s text_len=%d",
                file_token, len(text))
    body = {
        "file_type": file_type,
        "reply_elements": [
            {"type": "text", "text": text},
        ],
    }

    code, msg, _ = await _exec_request(
        client, "POST", _ADD_COMMENT_URI,
        paths={"file_token": file_token},
        body=body,
    )
    if code != 0:
        logger.warning("[Feishu-Comment] add_whole_comment FAILED: code=%s msg=%s", code, msg)
    else:
        logger.info("[Feishu-Comment] add_whole_comment OK")
    return code == 0


_REPLY_CHUNK_SIZE = 4000


def _chunk_text(text: str, limit: int = _REPLY_CHUNK_SIZE) -> List[str]:
    """Split text into chunks for delivery, preferring line breaks."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Find last newline within limit
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


async def deliver_comment_reply(
    client: Any,
    file_token: str,
    file_type: str,
    comment_id: str,
    text: str,
    is_whole: bool,
) -> bool:
    """Route agent reply to the correct API, chunking long text.

    - Whole comment -> add_whole_comment
    - Local comment -> reply_to_comment, fallback to add_whole_comment on 1069302
    """
    chunks = _chunk_text(text)
    logger.info("[Feishu-Comment] deliver_comment_reply: is_whole=%s comment_id=%s text_len=%d chunks=%d",
                is_whole, comment_id, len(text), len(chunks))

    all_ok = True
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info("[Feishu-Comment] deliver_comment_reply: sending chunk %d/%d (%d chars)",
                        i + 1, len(chunks), len(chunk))

        if is_whole:
            ok = await add_whole_comment(client, file_token, file_type, chunk)
        else:
            success, code = await reply_to_comment(client, file_token, file_type, comment_id, chunk)
            if success:
                ok = True
            elif code == 1069302:
                logger.info("[Feishu-Comment] Reply not allowed (1069302), falling back to add_whole_comment")
                ok = await add_whole_comment(client, file_token, file_type, chunk)
                is_whole = True  # subsequent chunks also use add_comment
            else:
                ok = False

        if not ok:
            all_ok = False
            break

    return all_ok


# ---------------------------------------------------------------------------
# Comment content extraction helpers
# ---------------------------------------------------------------------------


def _extract_reply_text(reply: Dict[str, Any]) -> str:
    """Extract plain text from a comment reply's content structure."""
    content = reply.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content

    elements = content.get("elements", [])
    parts = []
    for elem in elements:
        if elem.get("type") == "text_run":
            text_run = elem.get("text_run", {})
            parts.append(text_run.get("text", ""))
        elif elem.get("type") == "docs_link":
            docs_link = elem.get("docs_link", {})
            parts.append(docs_link.get("url", ""))
        elif elem.get("type") == "person":
            person = elem.get("person", {})
            parts.append(f"@{person.get('user_id', 'unknown')}")
    return "".join(parts)


def _get_reply_user_id(reply: Dict[str, Any]) -> str:
    """Extract user_id from a reply dict."""
    user_id = reply.get("user_id", "")
    if isinstance(user_id, dict):
        return user_id.get("open_id", "") or user_id.get("user_id", "")
    return str(user_id)


def _extract_semantic_text(reply: Dict[str, Any], self_open_id: str = "") -> str:
    """Extract semantic text from a reply, stripping self @mentions and extra whitespace."""
    content = reply.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content

    elements = content.get("elements", [])
    parts = []
    for elem in elements:
        if elem.get("type") == "person":
            person = elem.get("person", {})
            uid = person.get("user_id", "")
            # Skip self @mention (it's routing, not content)
            if self_open_id and uid == self_open_id:
                continue
            parts.append(f"@{uid}")
        elif elem.get("type") == "text_run":
            text_run = elem.get("text_run", {})
            parts.append(text_run.get("text", ""))
        elif elem.get("type") == "docs_link":
            docs_link = elem.get("docs_link", {})
            parts.append(docs_link.get("url", ""))
    return " ".join("".join(parts).split()).strip()


# ---------------------------------------------------------------------------
# Document link parsing and wiki resolution
# ---------------------------------------------------------------------------

import re as _re

# Matches feishu/lark document URLs and extracts doc_type + token
_FEISHU_DOC_URL_RE = _re.compile(
    r"(?:feishu\.cn|larkoffice\.com|larksuite\.com|lark\.suite\.com)"
    r"/(?P<doc_type>wiki|doc|docx|sheet|sheets|slides|mindnote|bitable|base|file)"
    r"/(?P<token>[A-Za-z0-9_-]{10,40})"
)

_WIKI_GET_NODE_URI = "/open-apis/wiki/v2/spaces/get_node"


def _extract_docs_links(replies: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract unique document links from a list of comment replies.

    Returns list of ``{"url": "...", "doc_type": "...", "token": "..."}`` dicts.
    """
    seen_tokens = set()
    links = []
    for reply in replies:
        content = reply.get("content", {})
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue
        for elem in content.get("elements", []):
            if elem.get("type") not in ("docs_link", "link"):
                continue
            link_data = elem.get("docs_link") or elem.get("link") or {}
            url = link_data.get("url", "")
            if not url:
                continue
            m = _FEISHU_DOC_URL_RE.search(url)
            if not m:
                continue
            doc_type = m.group("doc_type")
            token = m.group("token")
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            links.append({"url": url, "doc_type": doc_type, "token": token})
    return links


async def _reverse_lookup_wiki_token(
    client: Any, obj_type: str, obj_token: str,
) -> Optional[str]:
    """Reverse-lookup: given an obj_token, find its wiki node_token.

    Returns the wiki_token if the document belongs to a wiki space,
    or None if it doesn't or the API call fails.
    """
    code, msg, data = await _exec_request(
        client, "GET", _WIKI_GET_NODE_URI,
        queries=[("token", obj_token), ("obj_type", obj_type)],
    )
    if code == 0:
        node = data.get("node", {})
        wiki_token = node.get("node_token", "")
        return wiki_token if wiki_token else None
    # code != 0: either not a wiki doc or service error — log and return None
    logger.warning("[Feishu-Comment] Wiki reverse lookup failed: code=%s msg=%s obj=%s:%s", code, msg, obj_type, obj_token)
    return None


async def _resolve_wiki_nodes(
    client: Any,
    links: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Resolve wiki links to their underlying document type and token.

    Mutates entries in *links* in-place: replaces ``doc_type`` and ``token``
    with the resolved values for wiki links.  Non-wiki links are unchanged.
    """
    wiki_links = [l for l in links if l["doc_type"] == "wiki"]
    if not wiki_links:
        return links

    for link in wiki_links:
        wiki_token = link["token"]
        code, msg, data = await _exec_request(
            client, "GET", _WIKI_GET_NODE_URI,
            queries=[("token", wiki_token)],
        )
        if code == 0:
            node = data.get("node", {})
            resolved_type = node.get("obj_type", "")
            resolved_token = node.get("obj_token", "")
            if resolved_type and resolved_token:
                logger.info(
                    "[Feishu-Comment] Wiki resolved: %s -> %s:%s",
                    wiki_token, resolved_type, resolved_token,
                )
                link["resolved_type"] = resolved_type
                link["resolved_token"] = resolved_token
            else:
                logger.warning("[Feishu-Comment] Wiki resolve returned empty: %s", wiki_token)
        else:
            logger.warning("[Feishu-Comment] Wiki resolve failed: code=%s msg=%s token=%s", code, msg, wiki_token)

    return links


def _format_referenced_docs(
    links: List[Dict[str, str]], current_file_token: str = "",
) -> str:
    """Format resolved document links for prompt embedding."""
    if not links:
        return ""
    lines = ["", "Referenced documents in comments:"]
    for link in links:
        rtype = link.get("resolved_type", link["doc_type"])
        rtoken = link.get("resolved_token", link["token"])
        is_current = rtoken == current_file_token
        suffix = " (same as current document)" if is_current else ""
        lines.append(f"- {rtype}:{rtoken}{suffix} ({link['url'][:80]})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_PROMPT_TEXT_LIMIT = 220
_LOCAL_TIMELINE_LIMIT = 20
_WHOLE_TIMELINE_LIMIT = 12


def _truncate(text: str, limit: int = _PROMPT_TEXT_LIMIT) -> str:
    """Truncate text for prompt embedding."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _select_local_timeline(
    timeline: List[Tuple[str, str, bool]],
    target_index: int,
) -> List[Tuple[str, str, bool]]:
    """Select up to _LOCAL_TIMELINE_LIMIT entries centered on target_index.

    Always keeps first, target, and last entries.
    """
    if len(timeline) <= _LOCAL_TIMELINE_LIMIT:
        return timeline
    n = len(timeline)
    selected = set()
    selected.add(0)                            # first
    selected.add(n - 1)                        # last
    if 0 <= target_index < n:
        selected.add(target_index)             # current
    # Expand outward from target
    budget = _LOCAL_TIMELINE_LIMIT - len(selected)
    lo, hi = target_index - 1, target_index + 1
    while budget > 0 and (lo >= 0 or hi < n):
        if lo >= 0 and lo not in selected:
            selected.add(lo)
            budget -= 1
        lo -= 1
        if budget > 0 and hi < n and hi not in selected:
            selected.add(hi)
            budget -= 1
        hi += 1
    return [timeline[i] for i in sorted(selected)]


def _select_whole_timeline(
    timeline: List[Tuple[str, str, bool]],
    current_index: int,
    nearest_self_index: int,
) -> List[Tuple[str, str, bool]]:
    """Select up to _WHOLE_TIMELINE_LIMIT entries for whole-doc comments.

    Prioritizes current entry and nearest self reply.
    """
    if len(timeline) <= _WHOLE_TIMELINE_LIMIT:
        return timeline
    n = len(timeline)
    selected = set()
    if 0 <= current_index < n:
        selected.add(current_index)
    if 0 <= nearest_self_index < n:
        selected.add(nearest_self_index)
    # Expand outward from current
    budget = _WHOLE_TIMELINE_LIMIT - len(selected)
    lo, hi = current_index - 1, current_index + 1
    while budget > 0 and (lo >= 0 or hi < n):
        if lo >= 0 and lo not in selected:
            selected.add(lo)
            budget -= 1
        lo -= 1
        if budget > 0 and hi < n and hi not in selected:
            selected.add(hi)
            budget -= 1
        hi += 1
    if not selected:
        # Fallback: take last N entries
        return timeline[-_WHOLE_TIMELINE_LIMIT:]
    return [timeline[i] for i in sorted(selected)]


_COMMON_INSTRUCTIONS = """
This is a Feishu document comment thread, not an IM chat.
Your reply will be posted automatically. Just output the reply text.
Use the thread timeline above as the main context.
The quoted content is your primary anchor — insert/summarize/explain requests are about it.
Do not guess document content you haven't read.

If the quote, timeline, and referenced-document metadata above are enough,
output the final reply directly.

If you need the full text content of one or more documents to reply, output
exactly one line in this form (JSON object) and stop — do NOT include any
reply text in that response:
    <NEED_DOC_READ>{"tokens": ["<doc_token_1>", "<doc_token_2>"]}
You may only request tokens that appear in the "Current commented document"
section or the "Referenced documents from current user comment" section
above.  Non-docx or unknown tokens will be silently dropped.  The contents
will be fetched and you will be asked to reply again.

Reply in the same language as the user's comment unless they request otherwise.
Use plain text only. Do not use Markdown, headings, bullet lists, tables, or code blocks.
Do not show your reasoning process. Do not start with "I will", "Let me", or "I'll first".
Output only the final user-facing reply.
If no reply is needed, output exactly NO_REPLY.
""".strip()


def build_local_comment_prompt(
    *,
    doc_title: str,
    doc_url: str,
    file_token: str,
    file_type: str,
    comment_id: str,
    quote_text: str,
    root_comment_text: str,
    target_reply_text: str,
    timeline: List[Tuple[str, str, bool]],  # [(user_id, text, is_self)]
    self_open_id: str,
    target_index: int = -1,
    referenced_docs: str = "",
) -> str:
    """Build the prompt for a local (quoted-text) comment.

    All user-originated strings are passed through ``_strip_sentinel`` so a
    malicious commenter can't inject a forged ``<NEED_DOC_READ>`` marker.
    """
    selected = _select_local_timeline(timeline, target_index)

    lines = [
        f'The user added a reply in "{doc_title}".',
        f'Current user comment text: "{_truncate(_strip_sentinel(target_reply_text))}"',
        f'Original comment text: "{_truncate(_strip_sentinel(root_comment_text))}"',
        f'Quoted content: "{_truncate(_strip_sentinel(quote_text), 500)}"',
        "This comment mentioned you (@mention is for routing, not task content).",
        f"Document link: {doc_url}",
        "Current commented document:",
        f"- file_type={file_type}",
        f"- file_token={file_token}",
        f"- comment_id={comment_id}",
        "",
        f"Current comment card timeline ({len(selected)}/{len(timeline)} entries):",
    ]

    for user_id, text, is_self in selected:
        marker = " <-- YOU" if is_self else ""
        lines.append(f"[{user_id}] {_truncate(_strip_sentinel(text))}{marker}")

    if referenced_docs:
        lines.append(referenced_docs)

    lines.append("")
    lines.append(_COMMON_INSTRUCTIONS)
    return "\n".join(lines)


def build_whole_comment_prompt(
    *,
    doc_title: str,
    doc_url: str,
    file_token: str,
    file_type: str,
    comment_text: str,
    timeline: List[Tuple[str, str, bool]],  # [(user_id, text, is_self)]
    self_open_id: str,
    current_index: int = -1,
    nearest_self_index: int = -1,
    referenced_docs: str = "",
) -> str:
    """Build the prompt for a whole-document comment.

    All user-originated strings are passed through ``_strip_sentinel`` so a
    malicious commenter can't inject a forged ``<NEED_DOC_READ>`` marker.
    """
    selected = _select_whole_timeline(timeline, current_index, nearest_self_index)

    lines = [
        f'The user added a comment in "{doc_title}".',
        f'Current user comment text: "{_truncate(_strip_sentinel(comment_text))}"',
        "This is a whole-document comment.",
        "This comment mentioned you (@mention is for routing, not task content).",
        f"Document link: {doc_url}",
        "Current commented document:",
        f"- file_type={file_type}",
        f"- file_token={file_token}",
        "",
        f"Whole-document comment timeline ({len(selected)}/{len(timeline)} entries):",
    ]

    for user_id, text, is_self in selected:
        marker = " <-- YOU" if is_self else ""
        lines.append(f"[{user_id}] {_truncate(_strip_sentinel(text))}{marker}")

    if referenced_docs:
        lines.append(referenced_docs)

    lines.append("")
    lines.append(_COMMON_INSTRUCTIONS)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


def _resolve_model_and_runtime() -> Tuple[str, dict]:
    """Resolve model and provider credentials, same as gateway message handling."""
    import os
    from gateway.run import _load_gateway_config, _resolve_gateway_model

    user_config = _load_gateway_config()
    model = _resolve_gateway_model(user_config)

    from gateway.run import _resolve_runtime_agent_kwargs
    runtime_kwargs = _resolve_runtime_agent_kwargs()

    # Fall back to provider's default model if none configured
    if not model and runtime_kwargs.get("provider"):
        try:
            from hermes_cli.models import get_default_model_for_provider
            model = get_default_model_for_provider(runtime_kwargs["provider"])
        except Exception:
            pass

    return model, runtime_kwargs


# ---------------------------------------------------------------------------
# Session persistence (delegated to hermes's generic SessionStore)
#
# Comment sessions use ``chat_type="doc_comment"`` so they flow through the
# same SessionStore pipeline as IM — inheriting daily-reset, idle-reset,
# token tracking, and SQLite persistence automatically.
#
# Two scoping semantics, both keyed at the document level for ``chat_id``:
#
#   Local comment   thread_id = comment_id
#     key: agent:main:feishu:doc_comment:{file_type}:{file_token}:{comment_id}
#     Each comment thread (one root comment + its replies) is isolated.
#
#   Whole-doc       thread_id = _WHOLE_DOC_SENTINEL_THREAD_ID
#     key: agent:main:feishu:doc_comment:{file_type}:{file_token}:__whole_doc__
#     All whole-document comments on the same doc share one session —
#     matching the semantic that whole-doc comments form a document-level
#     discussion rather than per-thread conversations.
#
# user_id is deliberately not in the key: ``build_session_key`` skips
# per-user isolation when ``thread_id`` is truthy (under the default
# ``thread_sessions_per_user=False``), so we always route through that
# thread-shared branch — the sentinel for whole-doc is also truthy.
#
# Why cross-user sharing is safe here (different from IM):
#
#   IM's session-per-user model exists because DMs / threads in IM carry an
#   access boundary — user A's DM with the bot is not visible to user B.
#   Feishu document comments have the opposite property: whole-document
#   comments are inherently public to everyone with document access, and any
#   bot reply is likewise visible to every participant.  Collapsing all
#   whole-doc comments on a document onto one shared session therefore
#   mirrors the document's native visibility — it does not leak anything
#   across users, because nothing is private across users to begin with.
#
#   Consequently:
#
#   * No information-asymmetry risk.  A cannot "leak" to B via this session
#     because A's comments (and the bot's replies) are already visible to B
#     in the document itself.
#
#   * Cross-user context crosstalk (e.g. the agent treating A's "this part"
#     as B's referent) is a quality concern, not a privacy concern — the
#     prior context the agent reuses is the same content B can already see
#     on the document.  It also matches how whole-doc threads naturally
#     unfold: later participants pick up where the discussion left off.
#
#   * State churn on SessionEntry (token counts, memory_flushed,
#     updated_at) is serialized by SessionStore._lock for a given key, so
#     concurrent events from different users don't race on SessionEntry
#     fields — their effect on state ordering is equivalent to interleaved
#     turns from a single stream.
#
#   Local-comment threads are a different story and keep their per-
#   comment_id isolation above.
# ---------------------------------------------------------------------------

# Sentinel used as thread_id for whole-document comments.  Feishu comment_id
# values are alphanumeric strings (typically numeric), so a double-underscore
# literal never collides with a real thread.
_WHOLE_DOC_SENTINEL_THREAD_ID = "__whole_doc__"


def _build_comment_session_source(
    *,
    file_type: str,
    file_token: str,
    comment_id: str,
    is_whole_comment: bool,
    from_open_id: str,
    doc_title: Optional[str],
) -> "SessionSource":
    """Build the SessionSource identifying a comment-thread conversation.

    Whole-doc comments collapse to a single document-level session via the
    ``_WHOLE_DOC_SENTINEL_THREAD_ID`` sentinel; local comments stay
    isolated per-thread via their ``comment_id``.
    """
    from gateway.config import Platform
    from gateway.session import SessionSource

    # Prefer a readable display name; fall back to a short token-based stub
    # so session listings / logs remain greppable even without title info.
    display_name = doc_title or f"{file_type}:{file_token[:8]}"

    # Local: each comment card is its own session.
    # Whole-doc: all whole comments on the doc share one session.  Using a
    # constant sentinel (not None) keeps build_session_key on the
    # thread-shared branch, which is what prevents user_id from entering
    # the key under the default ``thread_sessions_per_user=False``.
    thread_id = (
        _WHOLE_DOC_SENTINEL_THREAD_ID if is_whole_comment else comment_id
    )

    return SessionSource(
        platform=Platform.FEISHU,
        chat_type="doc_comment",
        chat_id=f"{file_type}:{file_token}",
        chat_name=display_name,
        thread_id=thread_id,
        user_id=from_open_id,
    )


def _load_comment_history(
    session_store: Any, session_id: str,
) -> List[Dict[str, Any]]:
    """Load this session's prior transcript via the SessionStore public API.

    Goes through ``SessionStore.load_transcript`` rather than poking
    ``SessionStore._db`` directly.  This keeps doc_comment sessions on
    the same storage path as IM / other chat types — in particular:
      * JSONL is written in lockstep with SQLite (``append_to_transcript``)
      * reads are taken from whichever of JSONL / SQLite is longer, which
        guards against silent truncation when a session straddles the
        SessionDB introduction (see ``load_transcript`` in gateway/session.py)
      * future SessionStore evolutions (encryption, hooks, format changes)
        propagate automatically, instead of letting this path drift.

    Returns an empty list if the store raises for any reason — a flaky
    transcript layer must not crash the comment handler.
    """
    try:
        return session_store.load_transcript(session_id)
    except Exception as e:
        logger.warning(
            "[Feishu-Comment] failed to load history for session_id=%s: %s",
            session_id, e,
        )
        return []


def _persist_comment_turn(
    session_store: Any,
    session_id: str,
    user_prompt: str,
    assistant_reply: str,
) -> None:
    """Persist one user→assistant turn via the SessionStore public API.

    Same rationale as ``_load_comment_history``: route through
    ``append_to_transcript`` so doc_comment sessions share the storage
    semantics (SQLite + JSONL dual-write, future evolution, etc.) that
    IM and other chat types already rely on.

    Only final user-visible content is stored; the two-pass sentinel
    protocol's intermediate ``<NEED_DOC_READ>`` output and fetched-doc
    payload are per-turn mechanics, not durable dialogue state, so they
    are deliberately omitted to keep future prompts clean.
    """
    try:
        session_store.append_to_transcript(
            session_id, {"role": "user", "content": user_prompt},
        )
        session_store.append_to_transcript(
            session_id, {"role": "assistant", "content": assistant_reply},
        )
    except Exception as e:
        logger.warning(
            "[Feishu-Comment] failed to persist turn for session_id=%s: %s",
            session_id, e,
        )


# Upper bound for persisted user-turn size.  A single comment reply shouldn't
# be longer than this; abnormally long text is truncated with a marker so a
# runaway input can't bloat SessionDB rows indefinitely.
_MAX_PERSISTED_USER_TURN_CHARS = 2000


def _compact_user_turn_for_persistence(
    *,
    target_reply_text: str,
    quote_text: str = "",
) -> str:
    """Render a compact user-turn string suitable for SessionDB persistence.

    The live prompt built by ``build_local/whole_comment_prompt`` bundles
    timeline, referenced-doc metadata, and instruction boilerplate — all
    regenerated from fresh API data on each turn.  Persisting that full
    prompt would duplicate the evolving timeline into every historical row
    (O(n·k) bytes) and drown the transcript in repeated rules.

    This helper keeps only:
      * the user's actual comment text (``target_reply_text``)
      * an optional quote marker for local comments, so future turns can
        still resolve references like "this here" in history replay

    Both user-originated fields are passed through ``_strip_sentinel``
    before persistence.  Without this, a commenter could stash a
    ``<NEED_DOC_READ>`` literal in one turn and have it replayed
    unsanitized when the transcript becomes ``conversation_history`` on
    later turns — reopening the protocol-injection surface that the
    live-prompt path already closes.

    Result is clamped to ``_MAX_PERSISTED_USER_TURN_CHARS`` with a visible
    truncation marker — long paste events can't bloat the DB unboundedly.
    """
    parts: List[str] = []
    if quote_text:
        parts.append(f"[Quoted] {_strip_sentinel(quote_text)}")
    if target_reply_text:
        parts.append(_strip_sentinel(target_reply_text))
    out = "\n".join(parts)

    if len(out) > _MAX_PERSISTED_USER_TURN_CHARS:
        logger.warning(
            "[Feishu-Comment] persisted user turn truncated: %d → %d chars",
            len(out), _MAX_PERSISTED_USER_TURN_CHARS,
        )
        out = (
            out[:_MAX_PERSISTED_USER_TURN_CHARS]
            + f"\n[... truncated at {_MAX_PERSISTED_USER_TURN_CHARS} chars]"
        )
    return out


# ---------------------------------------------------------------------------
# Document-content fetch helpers (business-code equivalents of the v1
# ``feishu_doc_read`` tool).  Callable only from this module's two-pass
# agent orchestration — never exposed as agent tools.
# ---------------------------------------------------------------------------

_RAW_CONTENT_URI = "/open-apis/docx/v1/documents/:document_id/raw_content"

# Per-document and aggregate caps.  Feishu docs can run tens of thousands of
# characters; we truncate to keep the prompt within sane bounds.  The agent
# is told when truncation happened (see ``_format_doc_content_block``).
_MAX_DOC_CHARS = 30_000
_MAX_TOTAL_DOC_CHARS = 80_000


async def _read_document_raw_content(client: Any, document_id: str) -> str:
    """Fetch a Feishu docx's raw plain-text content.

    Raises ``RuntimeError`` on non-zero response codes so that the caller's
    ``asyncio.gather(return_exceptions=True)`` converts failures into
    exception objects that are rendered into prompt-visible error strings.
    """
    code, msg, data = await _exec_request(
        client,
        "GET",
        _RAW_CONTENT_URI,
        paths={"document_id": document_id},
    )
    if code != 0:
        raise RuntimeError(f"code={code} msg={msg}")
    return data.get("content", "") or ""


def _truncate_doc_content(content: str, token: str) -> str:
    """Truncate a single doc's content to ``_MAX_DOC_CHARS`` and annotate."""
    if len(content) <= _MAX_DOC_CHARS:
        return content
    logger.warning(
        "[Feishu-Comment] truncating doc %s from %d to %d chars",
        token, len(content), _MAX_DOC_CHARS,
    )
    return (
        content[:_MAX_DOC_CHARS]
        + f"\n\n[... truncated at {_MAX_DOC_CHARS} chars;"
        f" original length was {len(content)} chars]"
    )


def _enforce_total_doc_budget(
    contents: Dict[str, str],
) -> Dict[str, str]:
    """Scale each doc proportionally if the aggregate exceeds the total cap.

    Preserves per-token ordering.  Called after per-doc truncation, so this
    only kicks in when many docs are requested at once.
    """
    total = sum(len(c) for c in contents.values())
    if total <= _MAX_TOTAL_DOC_CHARS or total == 0:
        return contents
    ratio = _MAX_TOTAL_DOC_CHARS / total
    logger.warning(
        "[Feishu-Comment] aggregate doc content %d exceeds cap %d; scaling each by %.2f",
        total, _MAX_TOTAL_DOC_CHARS, ratio,
    )
    scaled: Dict[str, str] = {}
    for token, content in contents.items():
        cut = int(len(content) * ratio)
        scaled[token] = content[:cut] + "\n[... further truncated to fit aggregate cap]"
    return scaled


# Cap on concurrent raw_content fetches.  The whitelist already bounds N
# (docs must appear in the current comment context), so N is usually small;
# this semaphore is a cheap insurance against bursty fan-out triggering
# Feishu's per-app rate limit when a comment references many docs.
_DOC_FETCH_CONCURRENCY = 4


async def _fetch_docs_for_agent(
    client: Any, tokens: List[str],
) -> Dict[str, str]:
    """Fetch raw content for each token in parallel.

    Failures are captured as prompt-ready error strings so the second
    agent pass can still proceed (and the model knows which docs failed).
    Results are truncated individually and then collectively.

    Concurrency is capped at ``_DOC_FETCH_CONCURRENCY`` to stay friendly
    with Feishu's rate limits even if the whitelist admits many tokens.
    """
    if not tokens:
        return {}

    sem = asyncio.Semaphore(_DOC_FETCH_CONCURRENCY)

    async def _one(token: str) -> str:
        async with sem:
            try:
                content = await _read_document_raw_content(client, token)
            except Exception as e:
                logger.warning(
                    "[Feishu-Comment] doc fetch failed token=%s: %s",
                    token, e,
                )
                return f"[Failed to fetch this document: {type(e).__name__}: {e}]"
            return _truncate_doc_content(content, token)

    raw = await asyncio.gather(*[_one(t) for t in tokens])
    return _enforce_total_doc_budget(dict(zip(tokens, raw)))


# ---------------------------------------------------------------------------
# <NEED_DOC_READ> sentinel protocol
#
# Protocol: the first agent pass either outputs the final reply directly, or
# outputs exactly one line:
#     <NEED_DOC_READ>{"tokens": ["token_a", "token_b"]}
# Business code parses that line, fetches the requested (and whitelisted)
# docs, and runs a second pass with the content appended to the prompt.
# ---------------------------------------------------------------------------

# Matches the sentinel tag followed by a JSON object.  Uses DOTALL so embedded
# newlines inside the JSON (unlikely but legal) don't break the match.
_NEED_DOC_READ_PATTERN = re.compile(r"<NEED_DOC_READ>\s*(\{.*?\})", re.DOTALL)

# Any occurrence of the bare sentinel literal — used to neutralize it inside
# user-originated or fetched-doc text so an attacker can't forge the protocol
# marker from within prompt-embedded content.
_NEED_DOC_READ_LITERAL = re.compile(r"<NEED_DOC_READ>", re.IGNORECASE)
_SENTINEL_PLACEHOLDER = "<NEED_DOC_READ_STRIPPED>"


def _strip_sentinel(text: str) -> str:
    """Replace any ``<NEED_DOC_READ>`` literal inside untrusted text.

    Applied to every user-originated string (comment text, quotes, timeline
    entries) and to fetched document content before they are interpolated
    into a prompt.  This prevents an attacker who controls a comment or a
    whitelisted doc from forging the sentinel protocol marker.

    The replacement is a visible placeholder rather than an empty string so
    that the substitution is greppable in logs if something goes wrong.
    """
    if not text:
        return text
    return _NEED_DOC_READ_LITERAL.sub(_SENTINEL_PLACEHOLDER, text)


def _extract_effective_doc_token(link: Dict[str, Any]) -> Tuple[str, str]:
    """Return the (effective_type, effective_token) for a referenced doc link.

    After ``_resolve_wiki_nodes`` runs, wiki links carry ``resolved_type`` /
    ``resolved_token`` pointing to the real underlying doc.  Prefer those.
    Returns empty strings for links we couldn't resolve.
    """
    resolved_type = link.get("resolved_type") or ""
    resolved_token = link.get("resolved_token") or ""
    if resolved_type and resolved_token:
        return resolved_type, resolved_token
    return link.get("doc_type") or "", link.get("token") or ""


def _build_doc_token_whitelist(
    source_file_type: str,
    source_file_token: str,
    referenced_links: List[Dict[str, Any]],
) -> Set[str]:
    """Collect docx tokens that the agent is allowed to request for reading.

    Only docx is whitelisted — the raw-content API is docx-only.  The source
    document is whitelisted when it's docx; all referenced links that
    resolve to docx are added as well.
    """
    whitelist: Set[str] = set()
    if source_file_type == "docx" and source_file_token:
        whitelist.add(source_file_token)
    for link in referenced_links or []:
        eff_type, eff_token = _extract_effective_doc_token(link)
        if eff_type == "docx" and eff_token:
            whitelist.add(eff_token)
    return whitelist


@dataclass
class SentinelParseResult:
    """Structured outcome of ``<NEED_DOC_READ>`` sentinel parsing.

    Three mutually-exclusive states for the first-pass response:

    - ``has_sentinel=False``: no sentinel literal in the response at all.
      The response is the user-facing reply and should be delivered.
    - ``has_sentinel=True``, ``accepted_tokens`` non-empty: the agent
      asked for docs and the whitelist accepted at least one — the
      caller runs the second pass.
    - ``has_sentinel=True``, ``accepted_tokens`` empty: the agent emitted
      a sentinel, but its payload was malformed JSON, had no ``tokens``
      list, or every requested token was dropped by the whitelist.  The
      caller MUST NOT return the raw first-pass response to the user —
      it contains the sentinel literal, which is internal protocol
      plumbing and must never be delivered.
    """
    has_sentinel: bool
    accepted_tokens: List[str]


def _parse_need_doc_read_sentinel(
    response: str, whitelist: Set[str],
) -> SentinelParseResult:
    """Parse the first-pass response for a ``<NEED_DOC_READ>`` sentinel.

    Detection and extraction are deliberately split so that malformed
    variants of the marker (bare ``<NEED_DOC_READ>``, space-separated
    token lists, marker embedded in natural-language hedging, etc.)
    are still correctly identified as sentinel turns instead of being
    silently treated as the final reply.

    Two-step algorithm:

    1. **Detection** — ``_NEED_DOC_READ_LITERAL`` (broad, case-insensitive
       literal search).  If the marker appears anywhere in ``response``,
       this *is* a sentinel turn, no matter what follows.  Returning
       ``has_sentinel=False`` when the literal is present was the bug
       that allowed the marker to leak to the user as visible text.

    2. **Extraction** — ``_NEED_DOC_READ_PATTERN`` (strict, JSON-gated).
       Only used to pull out the ``{"tokens": [...]}`` payload.  A missing
       or malformed payload degrades to ``accepted_tokens=[]``, not to
       ``has_sentinel=False``.

    See ``SentinelParseResult`` for the three return states.  Non-whitelist
    (including non-docx) tokens are logged and dropped so business code
    only ever fetches docs the agent had advance knowledge of via the
    prompt metadata.
    """
    # Step 1: detection — literal present anywhere?
    if not _NEED_DOC_READ_LITERAL.search(response):
        return SentinelParseResult(has_sentinel=False, accepted_tokens=[])

    # Step 2: extraction — try to pull the JSON payload.
    match = _NEED_DOC_READ_PATTERN.search(response)
    if not match:
        logger.warning(
            "[Feishu-Comment] <NEED_DOC_READ> literal present but no JSON "
            "payload follows (len=%d); treating as no-token sentinel",
            len(response),
        )
        return SentinelParseResult(has_sentinel=True, accepted_tokens=[])

    payload_raw = match.group(1)
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError as e:
        logger.warning(
            "[Feishu-Comment] <NEED_DOC_READ> sentinel JSON parse failed: %s (payload=%r)",
            e, payload_raw[:200],
        )
        return SentinelParseResult(has_sentinel=True, accepted_tokens=[])
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        return SentinelParseResult(has_sentinel=True, accepted_tokens=[])

    accepted: List[str] = []
    rejected: List[str] = []
    for t in tokens:
        if isinstance(t, str) and t in whitelist:
            if t not in accepted:  # preserve order, drop duplicates
                accepted.append(t)
        else:
            rejected.append(t)
    if rejected:
        logger.warning(
            "[Feishu-Comment] Dropping %d tokens not in whitelist: %s",
            len(rejected), rejected,
        )
    return SentinelParseResult(has_sentinel=True, accepted_tokens=accepted)


def _format_doc_content_block(contents: Dict[str, str]) -> str:
    """Render fetched doc contents for injection into the second-pass prompt.

    Output deliberately forbids further ``<NEED_DOC_READ>`` on the second
    turn: we already gave the agent everything it asked for.  Looping would
    risk non-termination.

    Doc authors can edit doc content, so the fetched body is untrusted and
    passed through ``_strip_sentinel`` — this neutralizes forged protocol
    markers even though business code no longer parses the sentinel on the
    second pass (defense in depth).
    """
    lines = ["", "---", "Fetched document contents:"]
    for token, content in contents.items():
        lines.append("")
        lines.append(f"[Document token: {token}]")
        lines.append(_strip_sentinel(content))
    lines.append("")
    lines.append("---")
    lines.append(
        "Use the above content to generate the final reply. "
        "You MUST produce the final user-facing reply now. "
        "<NEED_DOC_READ> is no longer accepted in this turn."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Two-pass comment agent orchestration
# ---------------------------------------------------------------------------


def _build_comment_agent(runtime_kwargs: Dict[str, Any], model: str) -> Any:
    """Construct the AIAgent used for a single comment event.

    The agent is given **no feishu tools** — document content flows in via
    the ``<NEED_DOC_READ>`` sentinel protocol, not via tool calls.

    ``persist_session=False`` is critical: the durable transcript for a
    comment thread goes through ``SessionStore.append_to_transcript`` in
    *compact* form (user's actual reply text + optional quote anchor;
    see ``_compact_user_turn_for_persistence``).  With the default
    ``persist_session=True``, ``AIAgent._persist_session`` would in
    parallel write the helper agent's full in-memory message list to
    ``~/.hermes/logs/session_{id}.json`` and to its own SessionDB —
    re-leaking the first-pass rendered prompt (timeline, quote, doc URL)
    and the second-pass fetched document bodies that this module is
    explicitly designed to keep off durable storage.

    Known residual: ``AIAgent._save_session_log`` is also called directly
    from a handful of edge paths in ``run_agent.py`` (length continuation,
    certain retry failures) that bypass ``_persist_session`` and therefore
    ignore this flag.  Closing that gap requires a change in
    ``run_agent.py`` itself (making ``_save_session_log`` honour
    ``self.persist_session``) and is deliberately out of this module's
    scope.
    """
    from run_agent import AIAgent

    return AIAgent(
        model=model,
        base_url=runtime_kwargs.get("base_url"),
        api_key=runtime_kwargs.get("api_key"),
        provider=runtime_kwargs.get("provider"),
        api_mode=runtime_kwargs.get("api_mode"),
        credential_pool=runtime_kwargs.get("credential_pool"),
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        # No tools are enabled; the two-pass sentinel protocol gets at most
        # two model turns total (first pass + optional second pass after
        # doc-content injection), so a tiny iteration budget is enough.
        max_iterations=2,
        enabled_toolsets=[],
        persist_session=False,
    )


def _run_first_pass(
    agent: Any, prompt: str, history: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """First agent turn — returns (response_text, raw_result)."""
    logger.info(
        "[Feishu-Comment] first pass: prompt=%d chars, history=%d",
        len(prompt), len(history),
    )
    result = agent.run_conversation(
        prompt, conversation_history=history or None,
    )
    response = (result.get("final_response") or "").strip()
    # Response may contain user-visible reply text or the NEED_DOC_READ
    # sentinel; don't log its body.
    logger.info(
        "[Feishu-Comment] first pass done: api_calls=%d response_len=%d",
        result.get("api_calls", 0), len(response),
    )
    return response, result


def _run_second_pass(
    agent: Any,
    client: Any,
    requested_tokens: List[str],
    *,
    prior_history: List[Dict[str, Any]],
    first_prompt: str,
    first_response: str,
) -> Tuple[str, Dict[str, Any]]:
    """Fetch requested docs and run the second agent turn.

    ``AIAgent.run_conversation`` reinitializes its message list from
    ``conversation_history`` on every call, so reusing the same agent
    instance does NOT retain the first-pass context.  We explicitly
    rebuild history as:

        prior_history + [first_pass_user, first_pass_assistant] + doc_block

    so turn 2 still sees the user's original question, quote, timeline,
    and the agent's own prior ``<NEED_DOC_READ>`` line — the doc content
    block alone is not enough to reply coherently.
    """
    logger.info(
        "[Feishu-Comment] second pass: fetching %d docs: %s",
        len(requested_tokens), requested_tokens,
    )
    doc_contents = asyncio.run(_fetch_docs_for_agent(client, requested_tokens))
    doc_block = _format_doc_content_block(doc_contents)

    second_history: List[Dict[str, Any]] = list(prior_history) + [
        {"role": "user", "content": first_prompt},
        {"role": "assistant", "content": first_response},
    ]
    result = agent.run_conversation(
        doc_block, conversation_history=second_history,
    )
    response = (result.get("final_response") or "").strip()
    logger.info(
        "[Feishu-Comment] second pass done: api_calls=%d response_len=%d",
        result.get("api_calls", 0), len(response),
    )
    return response, result


def _run_comment_agent(
    prompt: str,
    client: Any,
    doc_token_whitelist: Set[str],
    history: List[Dict[str, Any]],
) -> str:
    """Run the comment agent with the two-pass sentinel protocol.

    Step 1: invoke the agent with ``prompt`` and the caller-provided
            ``history`` (already loaded from SessionStore).
    Step 2: parse the response for ``<NEED_DOC_READ>``.  If absent, the
            first-pass response is the final reply.
    Step 3: if present, fetch the whitelisted tokens and run a second
            agent turn with the doc contents appended.

    Persistence is the caller's responsibility — this function returns the
    final reply text (or empty string on failure) and leaves history I/O
    to ``handle_drive_comment_event`` which holds the ``SessionStore``.
    """
    try:
        model, runtime_kwargs = _resolve_model_and_runtime()
        logger.info(
            "[Feishu-Comment] _run_comment_agent: model=%s provider=%s base_url=%s history=%d",
            model, runtime_kwargs.get("provider"),
            (runtime_kwargs.get("base_url") or "")[:50], len(history),
        )

        agent = _build_comment_agent(runtime_kwargs, model)

        # First pass
        first_response, _ = _run_first_pass(agent, prompt, history)
        parse = _parse_need_doc_read_sentinel(first_response, doc_token_whitelist)

        if not parse.has_sentinel:
            # No sentinel at all — first-pass response IS the final reply.
            return first_response

        if not parse.accepted_tokens:
            # Sentinel was emitted, but its payload was malformed / had no
            # valid tokens / every token was dropped by the whitelist.  The
            # raw first-pass response contains the sentinel literal, which
            # is internal protocol text — it must NEVER be delivered to the
            # user.  Coerce to the NO_REPLY path by returning "".
            logger.warning(
                "[Feishu-Comment] first-pass emitted <NEED_DOC_READ> but no "
                "tokens survived parse/whitelist — dropping first-pass output "
                "to prevent protocol leak (response_len=%d)",
                len(first_response),
            )
            return ""

        # Second pass: doc contents requested.  Pass the first-pass prompt
        # and response so ``_run_second_pass`` can explicitly reconstruct
        # ``conversation_history`` — otherwise turn 2 loses the user's
        # original question, quote, and timeline.
        response, _ = _run_second_pass(
            agent, client, parse.accepted_tokens,
            prior_history=history,
            first_prompt=prompt,
            first_response=first_response,
        )
        return response

    except Exception as e:
        logger.exception("[Feishu-Comment] _run_comment_agent failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Event handler entry point
# ---------------------------------------------------------------------------

_NO_REPLY_SENTINEL = "NO_REPLY"


def _gate_outbound_reply(response: Optional[str]) -> Optional[str]:
    """Single gate for any text about to be posted back to Feishu.

    Invariant this function enforces:
        *The returned string, if non-None, never contains the*
        ``<NEED_DOC_READ>`` *literal and is not the* ``NO_REPLY`` *signal.*

    Every delivery path must funnel its agent-produced text through this
    gate.  Adding a new delivery path without calling it re-opens the
    leak window this module has closed repeatedly — the historical bugs
    all followed the pattern "spot a new malformed-marker variant, add a
    new regex check".  Centralising the guarantee as a single boolean
    invariant (literal-in-text → refuse) means every future variant is
    covered without adding another regex.

    Returns:
        The ``response`` unchanged when it is safe to deliver, or
        ``None`` meaning "skip delivery".  ``None`` is returned when:

        * the response is empty / whitespace only;
        * the response contains the ``NO_REPLY`` sentinel (agent chose
          not to reply);
        * the response contains the ``<NEED_DOC_READ>`` literal anywhere
          (a malformed sentinel escaped parsing, or the second pass
          ignored the "no more sentinel" instruction).  Refusing to
          deliver is strictly safer than shipping a half-formed marker
          to the comment thread.
    """
    if not response:
        return None
    stripped = response.strip()
    if not stripped:
        return None
    if _NO_REPLY_SENTINEL in response:
        return None
    if _NEED_DOC_READ_LITERAL.search(response):
        logger.error(
            "[Feishu-Comment] Refusing delivery: response contains "
            "<NEED_DOC_READ> literal (len=%d)",
            len(response),
        )
        return None
    return response


_ALLOWED_NOTICE_TYPES = {"add_comment", "add_reply"}


async def handle_drive_comment_event(
    ctx: CommentContext, data: Any,
) -> None:
    """Full orchestration for a drive comment event.

    *ctx* bundles the lark client, the gateway SessionStore, and the bot's
    own open_id (see ``CommentContext``).  The caller constructs it via
    ``CommentContext.from_adapter(adapter, self_open_id=...)``; the handler
    itself never touches adapter internals.

    1. Parse event + filter (self-reply, notice_type)
    2. Add OK reaction
    3. Fetch doc meta + comment details in parallel
    4. Branch on is_whole: build timeline
    5. Build prompt, run agent (history from SessionStore)
    6. Deliver reply + persist user/assistant turn to SessionStore
    """
    client = ctx.client
    session_store = ctx.session_store
    self_open_id = ctx.self_open_id

    logger.info("[Feishu-Comment] ========== handle_drive_comment_event START ==========")
    parsed = parse_drive_comment_event(data)
    if parsed is None:
        logger.warning("[Feishu-Comment] Dropping malformed drive comment event")
        return
    logger.info("[Feishu-Comment] [Step 0/5] Event parsed successfully")

    file_token = parsed["file_token"]
    file_type = parsed["file_type"]
    comment_id = parsed["comment_id"]
    reply_id = parsed["reply_id"]
    from_open_id = parsed["from_open_id"]
    to_open_id = parsed["to_open_id"]
    notice_type = parsed["notice_type"]
    is_mentioned = parsed["is_mentioned"]

    # Filter: self-reply, receiver check, notice_type, is_mentioned.
    if from_open_id and self_open_id and from_open_id == self_open_id:
        logger.debug("[Feishu-Comment] Skipping self-authored event: from=%s", from_open_id)
        return
    if not to_open_id or (self_open_id and to_open_id != self_open_id):
        logger.debug("[Feishu-Comment] Skipping event not addressed to self: to=%s", to_open_id or "(empty)")
        return
    if notice_type and notice_type not in _ALLOWED_NOTICE_TYPES:
        logger.debug("[Feishu-Comment] Skipping notice_type=%s", notice_type)
        return
    # ``is_mentioned`` is the authoritative signal that the user explicitly
    # @-ed the bot.  Without this gate, any comment on a document the bot
    # was previously invited to (and any reply in a thread the bot
    # participated in) would route here — a noisy, privacy-adverse
    # behavior.  Drop events that lack an explicit mention.
    if not is_mentioned:
        logger.debug(
            "[Feishu-Comment] Skipping unmentioned event: comment=%s from=%s",
            comment_id, from_open_id,
        )
        return
    if not file_token or not file_type or not comment_id:
        logger.warning("[Feishu-Comment] Missing required fields, skipping")
        return

    logger.info(
        "[Feishu-Comment] Event: notice=%s file=%s:%s comment=%s from=%s",
        notice_type, file_type, file_token, comment_id, from_open_id,
    )

    # Access control
    from gateway.platforms.feishu_comment_rules import load_config, resolve_rule, is_user_allowed, has_wiki_keys

    comments_cfg = load_config()
    rule = resolve_rule(comments_cfg, file_type, file_token)

    # If no exact match and config has wiki keys, try reverse-lookup
    if rule.match_source in ("wildcard", "top") and has_wiki_keys(comments_cfg):
        wiki_token = await _reverse_lookup_wiki_token(client, file_type, file_token)
        if wiki_token:
            rule = resolve_rule(comments_cfg, file_type, file_token, wiki_token=wiki_token)

    if not rule.enabled:
        logger.info("[Feishu-Comment] Comments disabled for %s:%s, skipping", file_type, file_token)
        return
    if not is_user_allowed(rule, from_open_id):
        logger.info("[Feishu-Comment] User %s denied (policy=%s, rule=%s)", from_open_id, rule.policy, rule.match_source)
        return

    logger.info("[Feishu-Comment] Access granted: user=%s policy=%s rule=%s", from_open_id, rule.policy, rule.match_source)
    if reply_id:
        asyncio.ensure_future(
            add_comment_reaction(
                client,
                file_token=file_token,
                file_type=file_type,
                reply_id=reply_id,
                reaction_type="OK",
            )
        )

    # Step 2: Parallel fetch -- doc meta + comment details
    logger.info("[Feishu-Comment] [Step 2/5] Parallel fetch: doc meta + comment batch_query")
    meta_task = asyncio.ensure_future(
        query_document_meta(client, file_token, file_type)
    )
    comment_task = asyncio.ensure_future(
        batch_query_comment(client, file_token, file_type, comment_id)
    )
    doc_meta, comment_detail = await asyncio.gather(meta_task, comment_task)

    doc_title = doc_meta.get("title", "Untitled")
    doc_url = doc_meta.get("url", "")
    is_whole = bool(comment_detail.get("is_whole"))

    logger.info(
        "[Feishu-Comment] Comment context: title=%s is_whole=%s",
        doc_title, is_whole,
    )

    # Step 3: Build timeline based on comment type
    logger.info("[Feishu-Comment] [Step 3/5] Building timeline (is_whole=%s)", is_whole)
    if is_whole:
        # Whole-document comment: fetch all whole comments as timeline
        logger.info("[Feishu-Comment] Fetching whole-document comments for timeline...")
        whole_comments = await list_whole_comments(client, file_token, file_type)

        timeline: List[Tuple[str, str, bool]] = []
        current_text = ""
        current_index = -1
        nearest_self_index = -1
        for wc in whole_comments:
            reply_list = wc.get("reply_list", {})
            if isinstance(reply_list, str):
                try:
                    reply_list = json.loads(reply_list)
                except (json.JSONDecodeError, TypeError):
                    reply_list = {}
            replies = reply_list.get("replies", [])
            for r in replies:
                uid = _get_reply_user_id(r)
                text = _extract_reply_text(r)
                is_self = (uid == self_open_id) if self_open_id else False
                idx = len(timeline)
                timeline.append((uid, text, is_self))
                if uid == from_open_id:
                    current_text = _extract_semantic_text(r, self_open_id)
                    current_index = idx
                if is_self:
                    nearest_self_index = idx

        if not current_text:
            for i, (uid, text, is_self) in reversed(list(enumerate(timeline))):
                if not is_self:
                    current_text = text
                    current_index = i
                    break

        # current_text is the user's comment; log indices and length only.
        logger.info("[Feishu-Comment] Whole timeline: %d entries, current_idx=%d, self_idx=%d, current_len=%d",
                    len(timeline), current_index, nearest_self_index,
                    len(current_text) if current_text else 0)

        # Extract and resolve document links from all replies
        all_raw_replies = []
        for wc in whole_comments:
            rl = wc.get("reply_list", {})
            if isinstance(rl, str):
                try:
                    rl = json.loads(rl)
                except (json.JSONDecodeError, TypeError):
                    rl = {}
            all_raw_replies.extend(rl.get("replies", []))
        doc_links = _extract_docs_links(all_raw_replies)
        if doc_links:
            doc_links = await _resolve_wiki_nodes(client, doc_links)
        ref_docs_text = _format_referenced_docs(doc_links, file_token)

        prompt = build_whole_comment_prompt(
            doc_title=doc_title,
            doc_url=doc_url,
            file_token=file_token,
            file_type=file_type,
            comment_text=current_text,
            timeline=timeline,
            self_open_id=self_open_id,
            current_index=current_index,
            nearest_self_index=nearest_self_index,
            referenced_docs=ref_docs_text,
        )
        # Persistence-only view: just the user's current whole-doc comment
        # text.  Whole-doc has no per-anchor quote, so the quote segment is
        # empty.
        persist_user_text = current_text
        persist_quote_text = ""

    else:
        # Local comment: fetch the comment thread replies
        logger.info("[Feishu-Comment] Fetching comment thread replies...")
        replies = await list_comment_replies(
            client, file_token, file_type, comment_id,
            expect_reply_id=reply_id,
        )

        quote_text = comment_detail.get("quote", "")

        timeline = []
        root_text = ""
        target_text = ""
        target_index = -1
        for i, r in enumerate(replies):
            uid = _get_reply_user_id(r)
            text = _extract_reply_text(r)
            is_self = (uid == self_open_id) if self_open_id else False
            timeline.append((uid, text, is_self))
            if i == 0:
                root_text = _extract_semantic_text(r, self_open_id)
            rid = r.get("reply_id", "")
            if rid and rid == reply_id:
                target_text = _extract_semantic_text(r, self_open_id)
                target_index = i

        if not target_text and timeline:
            for i, (uid, text, is_self) in reversed(list(enumerate(timeline))):
                if uid == from_open_id:
                    target_text = text
                    target_index = i
                    break

        # quote/root/target are user/agent content — log lengths only.
        logger.info("[Feishu-Comment] Local timeline: %d entries, target_idx=%d, "
                    "quote_len=%d root_len=%d target_len=%d",
                    len(timeline), target_index,
                    len(quote_text) if quote_text else 0,
                    len(root_text) if root_text else 0,
                    len(target_text) if target_text else 0)

        # Extract and resolve document links from replies
        doc_links = _extract_docs_links(replies)
        if doc_links:
            doc_links = await _resolve_wiki_nodes(client, doc_links)
        ref_docs_text = _format_referenced_docs(doc_links, file_token)

        prompt = build_local_comment_prompt(
            doc_title=doc_title,
            doc_url=doc_url,
            file_token=file_token,
            file_type=file_type,
            comment_id=comment_id,
            quote_text=quote_text,
            root_comment_text=root_text,
            target_reply_text=target_text,
            timeline=timeline,
            self_open_id=self_open_id,
            target_index=target_index,
            referenced_docs=ref_docs_text,
        )
        # Persistence-only view: the user's actual reply text plus the
        # quote they anchored to (if any).  Everything else in ``prompt``
        # (timeline, referenced-doc metadata, instructions) is rebuilt
        # from fresh API data on each turn and must not enter history.
        persist_user_text = target_text
        persist_quote_text = quote_text

    # Prompt contains the full quote + timeline — never log its content, even
    # at DEBUG, because agent.log's per-level threshold is configurable and a
    # misconfigured deployment would expose user comments to any log reader.
    logger.info("[Feishu-Comment] [Step 4/5] Prompt built (%d chars), running agent...", len(prompt))

    # Build the whitelist of document tokens the agent may request via the
    # <NEED_DOC_READ> sentinel: the source document (if docx) plus any
    # referenced docs resolved to docx.  Non-docx tokens cannot be read via
    # the raw_content API, so we never whitelist them.
    doc_token_whitelist = _build_doc_token_whitelist(
        file_type, file_token, doc_links,
    )

    # Resolve the per-comment-thread session via hermes's generic SessionStore.
    # Falls back to a stateless turn (history=[]) if the gateway didn't wire
    # a session_store into the adapter — this keeps tests and degraded
    # runtimes working without crashing.
    session_entry = None
    history: List[Dict[str, Any]] = []
    if session_store is not None:
        source = _build_comment_session_source(
            file_type=file_type,
            file_token=file_token,
            comment_id=comment_id,
            is_whole_comment=is_whole,
            from_open_id=from_open_id,
            doc_title=doc_title,
        )
        session_entry = session_store.get_or_create_session(source)
        history = _load_comment_history(session_store, session_entry.session_id)
        logger.info(
            "[Feishu-Comment] session resolved: key=%s id=%s history=%d",
            session_entry.session_key, session_entry.session_id, len(history),
        )
    else:
        logger.info(
            "[Feishu-Comment] no session_store on adapter — running stateless turn",
        )

    # Step 4: Run agent in a thread (run_conversation is synchronous).
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None, _run_comment_agent, prompt, client, doc_token_whitelist, history,
    )

    # Funnel the agent's response through the single outbound gate.  The
    # gate enforces the invariant "delivered text never contains the
    # <NEED_DOC_READ> literal and is not NO_REPLY" — see
    # ``_gate_outbound_reply`` for why this is the one place to check.
    delivery_text = _gate_outbound_reply(response)

    if delivery_text is None:
        logger.info("[Feishu-Comment] No reply delivered (empty / NO_REPLY / gated)")
    else:
        # Agent response is the final user-visible reply — log length only.
        logger.info("[Feishu-Comment] Agent response: %d chars", len(delivery_text))

        # Step 5: Deliver reply
        logger.info("[Feishu-Comment] [Step 5/5] Delivering reply (is_whole=%s, comment_id=%s)", is_whole, comment_id)
        success = await deliver_comment_reply(
            client, file_token, file_type, comment_id, delivery_text, is_whole,
        )
        if success:
            logger.info("[Feishu-Comment] Reply delivered successfully")
            # Persist only on successful delivery: a failed delivery means the
            # user never saw the reply, so treating it as "didn't happen" in
            # the transcript avoids confusing future turns.
            if session_entry is not None and session_store is not None:
                # Persist only the semantic payload, not the full rendered
                # prompt — see ``_compact_user_turn_for_persistence``.
                user_turn = _compact_user_turn_for_persistence(
                    target_reply_text=persist_user_text,
                    quote_text=persist_quote_text,
                )
                _persist_comment_turn(
                    session_store, session_entry.session_id,
                    user_prompt=user_turn, assistant_reply=delivery_text,
                )
        else:
            logger.error("[Feishu-Comment] Failed to deliver reply")

    # Cleanup: remove OK reaction (best-effort, fire-and-forget).
    # Mirrors the add-reaction call at the start of the handler — we don't
    # want this extra round-trip to Feishu to hold the event loop when the
    # reply has already been delivered.
    if reply_id:
        asyncio.ensure_future(
            delete_comment_reaction(
                client,
                file_token=file_token,
                file_type=file_type,
                reply_id=reply_id,
                reaction_type="OK",
            )
        )

    logger.info("[Feishu-Comment] ========== handle_drive_comment_event END ==========")
