import math
import csv
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


SCENE_PATH = "scene.xml"
OUTPUT_CSV = "calibration_results.csv"


@dataclass
class MotionResult:
    base_speed: float
    turn_speed: float
    t_forward: float
    t_turn: float
    forward_yaw_diff: float
    forward_lat_error: float
    forward_dist: float
    turn_yaw_err: float
    turn_pos_drift: float


def create_model():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    robot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    # 初期状態を保持
    qpos0 = np.copy(data.qpos)
    qvel0 = np.copy(data.qvel)
    return model, data, robot_bid, qpos0, qvel0


def reset_state(data: mujoco.MjData, qpos0: np.ndarray, qvel0: np.ndarray):
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(data.model, data)


def get_pose(data: mujoco.MjData, robot_bid: int):
    pos = np.array(data.xpos[robot_bid])
    rot = np.array(data.xmat[robot_bid]).reshape(3, 3)
    forward = rot[:, 0]
    yaw = math.degrees(math.atan2(forward[1], forward[0]))
    return pos, yaw


def wrap_deg(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def run_forward_test(model, data, robot_bid, qpos0, qvel0, base_speed: float, t_forward: float):
    reset_state(data, qpos0, qvel0)
    dt = model.opt.timestep
    steps = int(t_forward / dt)

    pos0, yaw0 = get_pose(data, robot_bid)

    # 左右同一速度で前進させる（actuator: motor_right, motor_left）
    for _ in range(steps):
        data.ctrl[0] = base_speed
        data.ctrl[1] = base_speed
        mujoco.mj_step(model, data)

    pos1, yaw1 = get_pose(data, robot_bid)
    dx, dy = pos1[0] - pos0[0], pos1[1] - pos0[1]

    # yaw0 方向を前方とした座標系での前後・左右成分
    yaw0_rad = math.radians(yaw0)
    fx, fy = math.cos(yaw0_rad), math.sin(yaw0_rad)
    lx, ly = -fy, fx  # 左向き

    fwd_dist = dx * fx + dy * fy
    lat_dist = dx * lx + dy * ly
    yaw_diff = wrap_deg(yaw1 - yaw0)

    return yaw_diff, lat_dist, fwd_dist


def run_turn_test(model, data, robot_bid, qpos0, qvel0, turn_speed: float, t_turn: float, target_angle: float):
    reset_state(data, qpos0, qvel0)
    dt = model.opt.timestep
    steps = int(t_turn / dt)

    pos0, yaw0 = get_pose(data, robot_bid)

    # 右旋回: 左前進・右後退
    data.ctrl[0] = turn_speed
    data.ctrl[1] = -turn_speed
    for _ in range(steps):
        mujoco.mj_step(model, data)

    pos1, yaw1 = get_pose(data, robot_bid)
    dx, dy = pos1[0] - pos0[0], pos1[1] - pos0[1]
    pos_drift = math.hypot(dx, dy)

    yaw_diff = wrap_deg(yaw1 - yaw0)
    yaw_err = wrap_deg(yaw_diff - target_angle)
    return yaw_err, pos_drift


def sweep_parameters():
    model, data, robot_bid, qpos0, qvel0 = create_model()

    base_speeds = [8.0, 10.0, 12.0, 15.0]
    turn_speeds = [6.0, 8.0, 10.0]
    t_forwards = [1.0, 1.5, 2.0]
    t_turns = [0.6, 0.8, 1.0]  # 90度ターン用

    results = []

    for bs in base_speeds:
        for ts in turn_speeds:
            for tf in t_forwards:
                for tt in t_turns:
                    f_yaw, f_lat, f_fwd = run_forward_test(model, data, robot_bid, qpos0, qvel0, bs, tf)
                    t_yaw_err, t_drift = run_turn_test(model, data, robot_bid, qpos0, qvel0, ts, tt, target_angle=90.0)

                    res = MotionResult(
                        base_speed=bs,
                        turn_speed=ts,
                        t_forward=tf,
                        t_turn=tt,
                        forward_yaw_diff=f_yaw,
                        forward_lat_error=f_lat,
                        forward_dist=f_fwd,
                        turn_yaw_err=t_yaw_err,
                        turn_pos_drift=t_drift,
                    )
                    results.append(res)

    return results


def write_results(results):
    path = Path(OUTPUT_CSV)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "base_speed",
                "turn_speed",
                "t_forward",
                "t_turn",
                "forward_yaw_diff",
                "forward_lat_error",
                "forward_dist",
                "turn_yaw_err",
                "turn_pos_drift",
                "score",
            ]
        )

        for r in results:
            # シンプルなスコア: 直進時のyawズレと横ズレを抑えつつ、前進距離を稼ぎ、
            # ターン時は目標角度に近く、位置ずれが小さいほど良い
            score = (
                abs(r.forward_yaw_diff)
                + abs(r.forward_lat_error) * 2.0
                - r.forward_dist
                + abs(r.turn_yaw_err) * 0.5
                + r.turn_pos_drift * 2.0
            )
            writer.writerow(
                [
                    r.base_speed,
                    r.turn_speed,
                    r.t_forward,
                    r.t_turn,
                    r.forward_yaw_diff,
                    r.forward_lat_error,
                    r.forward_dist,
                    r.turn_yaw_err,
                    r.turn_pos_drift,
                    score,
                ]
            )

    # ベストスコアをコンソールにも表示
    best = min(results, key=lambda r: abs(r.forward_yaw_diff) + abs(r.forward_lat_error) * 2.0 - r.forward_dist +
                                  abs(r.turn_yaw_err) * 0.5 + r.turn_pos_drift * 2.0)
    print("Best parameters (by heuristic score):")
    print(
        f"  base_speed={best.base_speed}, turn_speed={best.turn_speed}, "
        f"t_forward={best.t_forward}, t_turn={best.t_turn}"
    )


def main():
    print("Running motion calibration sweep (no AI)...")
    results = sweep_parameters()
    write_results(results)
    print(f"Calibration results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

