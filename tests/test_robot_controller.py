import json
from pathlib import Path

import numpy as np

from er_dashboard import RobotController, load_turn_calibration, CALIBRATION_FILE


def test_robot_controller_discrete_turn_uses_calibration():
    calib = {"speed": 8.0, "duration": 2.5}
    ctrl = RobotController(turn_calibration=calib)

    motors, duration = ctrl.decide_action(None, "turn_left_30")
    assert motors == [-8.0, 8.0]
    assert duration == 2.5

    motors, duration = ctrl.decide_action(None, "turn_right_30")
    assert motors == [8.0, -8.0]
    assert duration == 2.5


def test_robot_controller_approach_centered_speeds():
    ctrl = RobotController()
    # Centered box, moderate size -> should move forward with near-equal speeds
    bbox = [100, 450, 600, 550]  # center ~500
    motors, duration = ctrl.decide_action(bbox, "approach")
    assert duration > 0
    assert motors[0] > 0 and motors[1] > 0
    assert abs(motors[0] - motors[1]) < 0.5  # nearly straight


def test_load_turn_calibration_reads_file(tmp_path, monkeypatch):
    data = {"turn_speed": 9.0, "turn_duration": 3.3}
    calib_file = tmp_path / "calibration_turn.json"
    calib_file.write_text(json.dumps(data))

    monkeypatch.setattr("er_dashboard.CALIBRATION_FILE", calib_file)
    result = load_turn_calibration()
    assert result["speed"] == 9.0
    assert result["duration"] == 3.3


def test_load_turn_calibration_missing_returns_none(tmp_path, monkeypatch):
    calib_file = tmp_path / "missing.json"
    monkeypatch.setattr("er_dashboard.CALIBRATION_FILE", calib_file)
    assert load_turn_calibration() is None
