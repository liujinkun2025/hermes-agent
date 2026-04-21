"""Tests for the ``extra_tools=`` parameter on ``AIAgent.__init__``.

``extra_tools`` generalizes the existing memory-provider tool-injection
pattern into a first-class parameter — per-agent tool schemas are merged
into ``self.tools`` and their handlers are dispatched ahead of the global
registry.  These tests lock in the contract:

  * schemas merge into ``self.tools`` (model sees them)
  * handlers land in ``self._extra_tool_handlers`` (agent can dispatch)
  * ``_invoke_tool`` routes to extra handlers before falling through to
    the registry
  * built-in agent-level names (todo/memory/etc.) cannot be shadowed
  * empty / None input is a no-op
"""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    """Build minimal OpenAI-format tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _build_agent(extra_tools=None, base_tool_names=()):
    """Construct an AIAgent with mocked externals and a chosen registry view."""
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs(*base_tool_names),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            extra_tools=extra_tools,
        )
        agent.client = MagicMock()
        return agent


# ---------------------------------------------------------------------------
# Construction-time contract
# ---------------------------------------------------------------------------


class TestExtraToolsConstruction:
    """Schemas merge into the tool list visible to the model."""

    def test_none_is_noop(self):
        agent = _build_agent(extra_tools=None, base_tool_names=("web_search",))
        assert agent._extra_tool_handlers == {}
        assert {t["function"]["name"] for t in agent.tools} == {"web_search"}

    def test_empty_list_is_noop(self):
        agent = _build_agent(extra_tools=[], base_tool_names=("web_search",))
        assert agent._extra_tool_handlers == {}
        assert {t["function"]["name"] for t in agent.tools} == {"web_search"}

    def test_single_tool_merges_into_self_tools(self):
        schema = {
            "name": "feishu_doc_read",
            "description": "Read Feishu doc text",
            "parameters": {"type": "object", "properties": {"doc_token": {"type": "string"}}},
        }
        handler = MagicMock(return_value='{"content": "ok"}')
        agent = _build_agent(
            extra_tools=[{"schema": schema, "handler": handler}],
            base_tool_names=("web_search",),
        )

        # Schema visible to the model as a tool in self.tools.
        names = {t["function"]["name"] for t in agent.tools}
        assert names == {"web_search", "feishu_doc_read"}

        # Handler indexed on the instance.
        assert "feishu_doc_read" in agent._extra_tool_handlers
        assert agent._extra_tool_handlers["feishu_doc_read"] is handler

        # Name validation set updated so tool-call filtering accepts it.
        assert "feishu_doc_read" in agent.valid_tool_names

    def test_multiple_tools_all_merge(self):
        specs = [
            {"schema": {"name": f"domain_{i}", "description": "", "parameters": {}}, "handler": MagicMock()}
            for i in range(3)
        ]
        agent = _build_agent(extra_tools=specs, base_tool_names=())
        assert set(agent._extra_tool_handlers) == {"domain_0", "domain_1", "domain_2"}
        assert {t["function"]["name"] for t in agent.tools} == {"domain_0", "domain_1", "domain_2"}

    def test_schema_wrapped_in_type_function_envelope(self):
        """Merged schemas must match the OpenAI-format envelope used by the model API."""
        schema = {"name": "domain_x", "description": "", "parameters": {}}
        agent = _build_agent(
            extra_tools=[{"schema": schema, "handler": MagicMock()}],
            base_tool_names=(),
        )
        wrapped = next(t for t in agent.tools if t["function"]["name"] == "domain_x")
        assert wrapped == {"type": "function", "function": schema}


# ---------------------------------------------------------------------------
# Shadow prevention
# ---------------------------------------------------------------------------


class TestExtraToolsShadowPrevention:
    """extra_tools must not override tools already provided by the registry."""

    def test_name_collision_with_registry_is_ignored(self, caplog):
        """If extra_tools supplies a name already in self.tools, drop it."""
        handler = MagicMock()
        agent = _build_agent(
            extra_tools=[
                {
                    "schema": {"name": "web_search", "description": "", "parameters": {}},
                    "handler": handler,
                }
            ],
            base_tool_names=("web_search",),  # already in registry
        )
        # Registry version wins — handler never stored.
        assert "web_search" not in agent._extra_tool_handlers
        # self.tools still has exactly one web_search entry.
        web_search_count = sum(
            1 for t in agent.tools if t["function"]["name"] == "web_search"
        )
        assert web_search_count == 1

    def test_built_in_agent_tool_name_cannot_be_shadowed(self):
        """The hard-coded ``todo`` dispatch in _invoke_tool runs before extra."""
        handler = MagicMock()
        # Even if registry doesn't have "todo", the _invoke_tool dispatch
        # for "todo" is hard-coded and checked before extra_tools, so a
        # collision is harmless at dispatch time regardless of whether
        # we store the handler.  Verify the schema-merge side stays sane.
        agent = _build_agent(
            extra_tools=[
                {
                    "schema": {"name": "todo", "description": "", "parameters": {}},
                    "handler": handler,
                }
            ],
            base_tool_names=(),  # registry has no 'todo' in this mock
        )
        # Handler IS stored (registry didn't claim it), but _invoke_tool
        # dispatch order ensures the built-in path wins.  The dispatch
        # ordering is exercised in TestExtraToolsDispatch below.
        # Here we only verify schema merged without duplication.
        todo_count = sum(1 for t in agent.tools if t["function"]["name"] == "todo")
        assert todo_count == 1


# ---------------------------------------------------------------------------
# Dispatch contract
# ---------------------------------------------------------------------------


class TestExtraToolsDispatch:
    """``_invoke_tool`` routes to extra handlers before the registry."""

    def test_extra_tool_handler_invoked(self):
        handler = MagicMock(return_value='{"success": true}')
        agent = _build_agent(
            extra_tools=[
                {
                    "schema": {"name": "domain_x", "description": "", "parameters": {}},
                    "handler": handler,
                }
            ],
            base_tool_names=(),
        )

        result = agent._invoke_tool("domain_x", {"foo": "bar"}, effective_task_id="t1")
        handler.assert_called_once_with({"foo": "bar"})
        assert result == '{"success": true}'

    def test_unknown_name_falls_through_to_registry(self):
        """Names not in extra_tools must still hit ``handle_function_call``."""
        handler = MagicMock()
        agent = _build_agent(
            extra_tools=[
                {
                    "schema": {"name": "domain_x", "description": "", "parameters": {}},
                    "handler": handler,
                }
            ],
            base_tool_names=(),
        )

        with patch("run_agent.handle_function_call", return_value='{"from": "registry"}') as mock_reg:
            result = agent._invoke_tool("some_registry_tool", {}, effective_task_id="t1")
        handler.assert_not_called()
        mock_reg.assert_called_once()
        assert result == '{"from": "registry"}'

    def test_built_in_todo_not_shadowed_by_extra(self):
        """Even if extra_tools supplies a ``todo`` handler, the built-in wins.

        ``_invoke_tool`` dispatch order puts the hard-coded ``todo`` branch
        above the extra-tools branch, so the extra handler must not fire
        for that name.
        """
        extra_handler = MagicMock(return_value='{"from": "extra"}')
        agent = _build_agent(
            extra_tools=[
                {
                    "schema": {"name": "todo", "description": "", "parameters": {}},
                    "handler": extra_handler,
                }
            ],
            base_tool_names=(),
        )
        # ``todo_tool`` is lazy-imported inside _invoke_tool, so patch the
        # source module (not ``run_agent``).
        with patch("tools.todo_tool.todo_tool", return_value='{"from": "built-in"}'):
            result = agent._invoke_tool(
                "todo", {"todos": []}, effective_task_id="t1",
            )
        extra_handler.assert_not_called()
        assert result == '{"from": "built-in"}'


# ---------------------------------------------------------------------------
# End-to-end-ish contract: extra_tools advertised exactly like registry tools
# ---------------------------------------------------------------------------


class TestExtraToolsExposureSymmetric:
    """A model integrating with the agent should not be able to tell
    extra_tools apart from registry tools at the wire-protocol level."""

    def test_extra_tool_schema_survives_self_tools_shape(self):
        """extra_tools should land under the same ``{'type': 'function', ...}``
        envelope as registry-loaded tools."""
        schema = {
            "name": "image_generate",
            "description": "Generate an image",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
        }
        agent = _build_agent(
            extra_tools=[{"schema": schema, "handler": MagicMock()}],
            base_tool_names=("web_search",),
        )
        registry_item = next(t for t in agent.tools if t["function"]["name"] == "web_search")
        extra_item = next(t for t in agent.tools if t["function"]["name"] == "image_generate")
        assert registry_item.keys() == extra_item.keys()
        assert registry_item["type"] == extra_item["type"] == "function"
