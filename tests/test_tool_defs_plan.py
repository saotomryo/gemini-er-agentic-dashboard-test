import json

from src.tool_defs import TOOL_DEFS, list_tool_names


def test_tool_defs_include_move_forward_long():
    names = list_tool_names()
    assert "move_forward_long" in names


def test_tool_defs_json_serializable():
    # Ensure tool defs are JSON-serializable for prompts
    json_str = json.dumps(TOOL_DEFS)
    assert "scan" in json_str
