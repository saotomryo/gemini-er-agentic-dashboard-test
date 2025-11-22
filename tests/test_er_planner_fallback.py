import numpy as np
import pytest

from er_dashboard import ERPlannerAssist


class DummyResp:
    def __init__(self, text):
        self.text = text


class DummyModel:
    def __init__(self, text):
        self._text = text
        self.called_with = None

    def generate_content(self, contents, generation_config=None):
        # Keep payload for inspection if needed
        self.called_with = (contents, generation_config)
        return DummyResp(self._text)


def test_er_fallback_returns_json(monkeypatch):
    """ERPlannerAssist should parse JSON from the model response."""
    planner = ERPlannerAssist.__new__(ERPlannerAssist)  # bypass __init__
    planner.model_name = "dummy"
    planner.model = DummyModel('{"action":"scan","reason":"fallback"}')

    step = planner.plan("Go to the pole", ["scan"], np.zeros((10, 10, 3), dtype=np.uint8), None)
    assert step["action"] == "scan"
    assert step["reason"] == "fallback"


def test_er_fallback_raises_none_on_error(monkeypatch):
    """ERPlannerAssist should return None if the model raises."""
    class BadModel:
        def generate_content(self, *args, **kwargs):
            raise RuntimeError("fail")

    planner = ERPlannerAssist.__new__(ERPlannerAssist)
    planner.model_name = "dummy"
    planner.model = BadModel()

    step = planner.plan("Go to the pole", [], np.zeros((1, 1, 3), dtype=np.uint8), None)
    assert step is None
