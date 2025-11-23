"""
Tool definitions for future ER tool-calling integration.

Each tool maps to an existing executor action in the dashboard:
- scan -> triggers the 360 scan routine
- turn_left / turn_right -> mapped to turn_left_30 / turn_right_30 with angle param (degrees)
- move_forward -> mapped to move_forward_short with duration param (seconds)
- approach -> mapped to approach with target
- stop -> mapped to stop
"""

TOOL_DEFS = [
    {
        "name": "scan",
        "description": "Rotate in place and capture 4 surround images (front/right/back/left).",
        "parameters": [],
    },
    {
        "name": "turn_left",
        "description": "Turn left by a given angle in degrees (approximate).",
        "parameters": [
            {"name": "angle_deg", "type": "number", "description": "Angle to turn left"},
        ],
    },
    {
        "name": "turn_right",
        "description": "Turn right by a given angle in degrees (approximate).",
        "parameters": [
            {"name": "angle_deg", "type": "number", "description": "Angle to turn right"},
        ],
    },
    {
        "name": "move_forward",
        "description": "Move forward for a duration (seconds).",
        "parameters": [
            {"name": "duration", "type": "number", "description": "Duration in seconds"},
        ],
    },
    {
        "name": "move_forward_long",
        "description": "Move forward for a longer duration (seconds), used when target is far and centered.",
        "parameters": [
            {"name": "duration", "type": "number", "description": "Duration in seconds"},
        ],
    },
    {
        "name": "approach",
        "description": "Use visual servoing to approach a visible target.",
        "parameters": [
            {"name": "target", "type": "string", "description": "Target name, e.g., red pole"},
        ],
    },
    {
        "name": "stop",
        "description": "Stop all motion.",
        "parameters": [],
    },
]


def list_tool_names():
    return [t["name"] for t in TOOL_DEFS]
