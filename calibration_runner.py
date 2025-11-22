import json
from pathlib import Path
from dataclasses import dataclass

import mujoco
import numpy as np

SCENE_PATH = "scene_calibration.xml"
DT = 0.002
CALIB_OUTPUT = Path("calibration_turn.json")


@dataclass
class TurnResult:
    direction: str
    speed: float
    duration: float
    yaw_delta: float
    pos_delta: tuple[float, float]


def get_yaw(model, data) -> float:
    """Compute yaw in degrees from the robot body pose."""
    robot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    rot = np.array(data.xmat[robot_bid]).reshape(3, 3)
    forward = rot[:, 0]
    return float(np.degrees(np.arctan2(forward[1], forward[0])))


def reset_state(model, data):
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0


def simulate_turn(model, data, left, right, duration_s) -> TurnResult:
    robot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")

    yaw0 = get_yaw(model, data)
    pos0 = np.array(data.xpos[robot_bid])

    steps = int(duration_s / DT)
    for _ in range(steps):
        data.ctrl[0] = right
        data.ctrl[1] = left
        mujoco.mj_step(model, data)

    yaw1 = get_yaw(model, data)
    pos1 = np.array(data.xpos[robot_bid])

    delta_yaw = ((yaw1 - yaw0 + 180) % 360) - 180  # wrap to [-180, 180]
    delta_pos = (pos1 - pos0)[:2]

    direction = "left" if left < 0 < right else "right"
    return TurnResult(direction, abs(left), duration_s, delta_yaw, (delta_pos[0], delta_pos[1]))


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    speeds = [3.0, 4.0, 5.0, 6.0, 7.0]
    durations = [0.8, 1.0, 1.3, 1.6, 2.0]

    results: list[TurnResult] = []

    for speed in speeds:
        for dur in durations:
            # Left turn
            reset_state(model, data)
            r = simulate_turn(model, data, left=-speed, right=speed, duration_s=dur)
            results.append(r)

            # Right turn
            reset_state(model, data)
            r = simulate_turn(model, data, left=speed, right=-speed, duration_s=dur)
            results.append(r)

    print("=== Turn Calibration (no AI) ===")
    print(f"Scene: {SCENE_PATH}, timestep={DT}s")
    for r in results:
        print(
            f"{r.direction:5s} speed={r.speed:4.1f} dur={r.duration:3.1f}s "
            f"yawΔ={r.yaw_delta:+6.1f} deg posΔ=({r.pos_delta[0]:+.2f},{r.pos_delta[1]:+.2f})"
        )

    # Simple suggestion: pick closest to 30deg
    target = 30.0
    best = sorted(results, key=lambda x: abs(abs(x.yaw_delta) - target))[0]
    print("\nSuggested ~30deg turn parameters:")
    print(
        f"direction={best.direction}, speed={best.speed:.1f}, duration={best.duration:.1f}s "
        f"(yawΔ={best.yaw_delta:+.1f} deg)"
    )

    payload = {
        "turn_speed": best.speed,
        "turn_duration": best.duration,
        "measured_yaw_delta": best.yaw_delta,
        "target_yaw": target,
        "scene": SCENE_PATH,
    }
    CALIB_OUTPUT.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved calibration to {CALIB_OUTPUT}")


if __name__ == "__main__":
    main()
