from tool_defs import TOOL_DEFS, list_tool_names


def test_tool_defs_contains_expected():
    names = list_tool_names()
    assert "scan" in names
    assert "turn_left" in names
    assert "turn_right" in names
    assert "move_forward" in names
    assert "approach" in names
    assert "stop" in names


def test_tool_defs_have_parameters():
    for tool in TOOL_DEFS:
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
