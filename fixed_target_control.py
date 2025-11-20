import math
import time

import cv2
import mujoco
import numpy as np


SCENE_PATH = "scene.xml"

# キャリブレーション結果に基づく基本パラメータ
BASE_SPEED = 15.0    # キャリブレーション済みの基準前進速度
TURN_SPEED = 6.0     # キャリブレーション済みの基準旋回速度
SPEED_GAIN = 1.5     # 速度ゲイン（1.5 倍）
ALIGN_THRESHOLD_DEG = 10.0  # これ以上ズレていたらその場旋回
REACH_DISTANCE = 0.5        # 目標までの到達距離 [m]


def wrap_deg(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def get_pose(model: mujoco.MjModel, data: mujoco.MjData, body_id: int):
    pos = np.array(data.xpos[body_id])
    rot = np.array(data.xmat[body_id]).reshape(3, 3)
    # このモデルではロボットの「前向き」は body の y 軸方向なので、
    # 回転行列の第2列（index=1）を前方ベクトルとして使用する。
    forward = rot[:, 1]
    yaw = math.degrees(math.atan2(forward[1], forward[0]))
    return pos, yaw


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    robot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_red")

    # --- 初期配置の明示設定 ---
    # ロボットは scene.xml のデフォルト位置 (0,0) 付近にそのまま置き、
    # 赤いポールをロボット前方の少し離れた位置に移動する。
    def set_pole_xy(joint_x_name: str, joint_y_name: str, x: float, y: float):
        try:
            jx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_x_name)
            jy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_y_name)
        except Exception:
            return
        adr_x = model.jnt_qposadr[jx]
        adr_y = model.jnt_qposadr[jy]
        data.qpos[adr_x] = x
        data.qpos[adr_y] = y

    def randomize_red_pole_around_robot(radius_min: float = 2.0, radius_max: float = 4.0):
        """ロボット位置を中心に、一定距離離れた位置に赤ポールをランダム生成"""
        robot_pos, _ = get_pose(model, data, robot_bid)
        r = np.random.uniform(radius_min, radius_max)
        theta = np.random.uniform(0.0, 2.0 * math.pi)
        x = robot_pos[0] + r * math.cos(theta)
        y = robot_pos[1] + r * math.sin(theta)
        set_pole_xy("joint_target_red_x", "joint_target_red_y", x, y)
        print(f"[SPAWN] red pole at ({x:.2f}, {y:.2f}) (r={r:.2f})")

    # 赤ポールをロボット周囲のランダムな位置に配置
    randomize_red_pole_around_robot()
    # 他のポールは遠くに退避させて、干渉しないようにする
    set_pole_xy("joint_target_blue_x", "joint_target_blue_y", -10.0, -10.0)
    set_pole_xy("joint_target_yellow_x", "joint_target_yellow_y", 10.0, -10.0)
    mujoco.mj_forward(model, data)

    # レンダラー（全体視点だけ表示）
    renderer = mujoco.Renderer(model, height=480, width=640)

    dt = model.opt.timestep

    # 実際に使う速度（ゲイン込み）
    fwd_speed = BASE_SPEED * SPEED_GAIN
    turn_speed = TURN_SPEED * SPEED_GAIN

    print("=== Fixed Target Control (no AI) ===")
    print("Robot will rotate to face RED pole (random spawn), then move straight.")

    step = 0
    while True:
        # 現在のロボット姿勢とターゲット位置を取得
        robot_pos, robot_yaw = get_pose(model, data, robot_bid)
        target_pos, _ = get_pose(model, data, target_bid)

        # y 方向の符号を反転させて、「画面下方向」を正面の前方として扱う
        # 画像座標系との対応を合わせるため、x 方向もロボット視点から見た符号に揃える
        dx = robot_pos[0] - target_pos[0]
        dy = robot_pos[1] - target_pos[1]
        dist = math.hypot(dx, dy)
        target_bearing = math.degrees(math.atan2(dy, dx))
        heading_err = wrap_deg(target_bearing - robot_yaw)

        # 制御ロジック
        if dist < REACH_DISTANCE:
            # 目標に到達したら赤ポールを別の位置に移動し、再チャレンジ
            ctrl_left = 0.0
            ctrl_right = 0.0
            print(f"[STEP {step}] REACHED: dist={dist:.2f} yaw={robot_yaw:.1f} -> respawn target")
            randomize_red_pole_around_robot()
        else:
            if abs(heading_err) > ALIGN_THRESHOLD_DEG:
                # まずは向きを合わせる
                # heading_err > 0: ターゲットが左側 -> 左に回る
                # heading_err < 0: ターゲットが右側 -> 右に回る
                turn = turn_speed
                if heading_err > 0:
                    ctrl_left = turn
                    ctrl_right = -turn
                    mode = "turn_left"
                else:
                    ctrl_left = -turn
                    ctrl_right = turn
                    mode = "turn_right"
            else:
                # ほぼ正面 -> 前進
                ctrl_left = fwd_speed
                ctrl_right = fwd_speed
                mode = "forward"

            print(
                f"[STEP {step}] mode={mode} dist={dist:.2f} "
                f"heading_err={heading_err:+.1f} yaw={robot_yaw:+.1f} "
                f"ctrl=({ctrl_left:+.1f},{ctrl_right:+.1f})"
            )

        # MuJoCo への指令適用（右が ctrl[0], 左が ctrl[1]）
        data.ctrl[0] = ctrl_right
        data.ctrl[1] = ctrl_left
        mujoco.mj_step(model, data)
        step += 1

        # 画面表示
        renderer.update_scene(data, camera="global_cam")
        img = renderer.render()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Global View (Fixed Target Mode)", img_bgr)
        if cv2.waitKey(1) == 27:
            break

        # 少し待つ（見やすさ用）
        time.sleep(dt)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
