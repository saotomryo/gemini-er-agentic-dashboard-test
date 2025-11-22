import numpy as np

from er_dashboard import ERProgressMonitor


class DummyResp:
    def __init__(self, text):
        self.text = text


class DummyModel:
    def __init__(self, text):
        self.text = text

    def generate_content(self, contents, generation_config=None):
        return DummyResp(self.text)


def test_progress_monitor_parses_json():
    mon = ERProgressMonitor.__new__(ERProgressMonitor)  # bypass __init__
    mon.model = DummyModel('{"status":"IN_PROGRESS","progress":0.5,"comment":"ok"}')
    result = mon.estimate("instr", [], [np.zeros((2, 2, 3), dtype=np.uint8)])
    assert result["status"] == "IN_PROGRESS"
    assert result["progress"] == 0.5


def test_progress_monitor_returns_none_on_error():
    class BadModel:
        def generate_content(self, *args, **kwargs):
            raise RuntimeError("fail")
    mon = ERProgressMonitor.__new__(ERProgressMonitor)
    mon.model = BadModel()
    result = mon.estimate("instr", [], [np.zeros((1, 1, 3), dtype=np.uint8)])
    assert result is None
