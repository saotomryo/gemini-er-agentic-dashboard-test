import numpy as np

from er_dashboard import ERMapper


class DummyResp:
    def __init__(self, text):
        self.text = text


class DummyModel:
    def __init__(self, text):
        self.text = text

    def generate_content(self, contents, generation_config=None):
        return DummyResp(self.text)


def test_er_mapper_returns_json():
    mapper = ERMapper.__new__(ERMapper)  # bypass __init__
    mapper.model_name = "dummy"
    mapper.model = DummyModel('{"targets": [], "map": [], "reason": "test"}')
    res = mapper.estimate_map("instr", [], np.zeros((2, 2, 3), dtype=np.uint8), None, {})
    assert res["map"] == []
    assert res["reason"] == "test"


def test_er_mapper_returns_none_on_error():
    class BadModel:
        def generate_content(self, *args, **kwargs):
            raise RuntimeError("fail")
    mapper = ERMapper.__new__(ERMapper)
    mapper.model_name = "dummy"
    mapper.model = BadModel()
    res = mapper.estimate_map("instr", [], np.zeros((1, 1, 3), dtype=np.uint8), None, {})
    assert res is None
