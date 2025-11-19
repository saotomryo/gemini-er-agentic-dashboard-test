import time
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai


# ==========================================
# 環境・エージェント設定
# ==========================================

SCENE_PATH = "scene.xml"

# 軽量モデルでスタート（後で ER に差し替え）
VISION_MODEL_NAME = "gemini-1.5-flash"


@dataclass
class RobotState:
    """ロボットの状態表示用の簡易データ構造"""

    position: np.ndarray  # world 座標 (x, y, z)
    yaw_deg: float        # 水平面での向き [deg]
    ctrl_left: float      # 左モーター指令値
    ctrl_right: float     # 右モーター指令値


class SimulationEnv:
    """MuJoCo 環境の薄いラッパ"""

    def __init__(self, xml_path: str = SCENE_PATH):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # カメラ
        self.robot_renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.global_renderer = mujoco.Renderer(self.model, height=480, width=640)

        # ロボット body の ID（姿勢取得用）
        self.robot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")

    def step(self, ctrl_left: float, ctrl_right: float):
        """1 ステップだけ物理シミュレーションを進める"""
        # actuator の並び: motor_right, motor_left
        # data.ctrl[0] -> 右, data.ctrl[1] -> 左
        self.data.ctrl[0] = ctrl_right
        self.data.ctrl[1] = ctrl_left

        mujoco.mj_step(self.model, self.data)

    def get_images(self):
        """ロボット視点・俯瞰視点の RGB 画像を返す"""
        # ロボット視点
        self.robot_renderer.update_scene(self.data, camera="robot_cam")
        img_robot = self.robot_renderer.render()

        # 俯瞰視点
        self.global_renderer.update_scene(self.data, camera="global_cam")
        img_global = self.global_renderer.render()

        return img_robot, img_global

    def get_robot_state(self, ctrl_left: float, ctrl_right: float) -> RobotState:
        """MuJoCo の状態からロボットの位置と向きを取得"""
        # 位置 (world 座標)
        pos = np.array(self.data.xpos[self.robot_bid])

        # 向き: xmat (3x3 回転行列) から yaw を推定
        rot = np.array(self.data.xmat[self.robot_bid]).reshape(3, 3)
        # 前方ベクトルは body x 軸と仮定
        forward = rot[:, 0]
        yaw = np.degrees(np.arctan2(forward[1], forward[0]))

        return RobotState(position=pos, yaw_deg=yaw, ctrl_left=ctrl_left, ctrl_right=ctrl_right)


class VisionSystem:
    """軽量 Gemini モデルを使った物体検出（後で ER に差し替え可能）"""

    def __init__(self, model_name: str = VISION_MODEL_NAME):
        load_dotenv()
        genai.configure()
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)

    def detect_object(self, img_rgb: np.ndarray, target_description: str):
        """画像とターゲット説明から 2D バウンディングボックスを取得

        返り値:
            [ymin, xmin, ymax, xmax] (0-1000 正規化) または None
        """
        from PIL import Image
        import json

        prompt = f"""
        Find the "{target_description}" in the image.
        Return JSON with key "box_2d" [ymin, xmin, ymax, xmax], values in 0-1000.
        If not found, return null.
        """

        try:
            pil_img = Image.fromarray(img_rgb)
            response = self.model.generate_content(
                [prompt, pil_img],
                generation_config={"response_mime_type": "application/json"},
            )
            data = json.loads(response.text)
            if "box_2d" in data and data["box_2d"]:
                return data["box_2d"]
        except Exception:
            pass
        return None


class DashboardUI:
    """俯瞰画像・ロボット視点・テキスト情報を 1 画面にまとめて表示する"""

    def __init__(self, window_name: str = "Robot Dashboard"):
        self.window_name = window_name

        # キャンバスサイズ（上: 画像, 下: 情報）
        self.canvas_h = 720
        self.canvas_w = 960

    def render(
        self,
        img_robot: np.ndarray,
        img_global: np.ndarray,
        robot_state: RobotState,
        user_instruction: str,
        current_task: str,
        task_progress: str,
        model_name: str,
    ):
        # BGR に変換（OpenCV 用）
        robot_bgr = cv2.cvtColor(img_robot, cv2.COLOR_RGB2BGR)
        global_bgr = cv2.cvtColor(img_global, cv2.COLOR_RGB2BGR)

        # 表示用にサイズ調整
        global_view = cv2.resize(global_bgr, (640, 480))
        robot_view = cv2.resize(robot_bgr, (320, 240))
        robot_view = cv2.resize(robot_view, (320, 480))

        # キャンバス作成（黒）
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        # 上段: 俯瞰 (左) + ロボ視点 (右)
        canvas[0:480, 0:640] = global_view
        canvas[0:480, 640:960] = robot_view

        # 下段テキスト領域
        y_base = 500
        line_h = 24

        # 左側: テキスト指示・タスク
        cv2.putText(canvas, "Instruction:", (10, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(canvas, user_instruction, (10, y_base + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(canvas, "Current Task:", (10, y_base + 2 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(canvas, current_task, (10, y_base + 3 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(canvas, "Progress:", (10, y_base + 4 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(canvas, task_progress, (10, y_base + 5 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 右側: ロボット状態・モデル
        x_right = 650
        pos = robot_state.position
        cv2.putText(canvas, "Robot State", (x_right, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)
        cv2.putText(
            canvas,
            f"pos: ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})",
            (x_right, y_base + line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            canvas,
            f"yaw: {robot_state.yaw_deg:+.1f} deg",
            (x_right, y_base + 2 * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            canvas,
            f"ctrl L/R: {robot_state.ctrl_left:+.1f}, {robot_state.ctrl_right:+.1f}",
            (x_right, y_base + 3 * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.putText(canvas, "Vision Model:", (x_right, y_base + 5 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(canvas, model_name, (x_right, y_base + 6 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(self.window_name, canvas)


def main():
    env = SimulationEnv()
    ui = DashboardUI()

    # 今はダミーの制御値で回しつつ、ダッシュボードだけ確認する
    ctrl_left = 0.0
    ctrl_right = 0.0

    user_instruction = "デモ: ダッシュボード表示のみ"
    current_task = "N/A"
    task_progress = "0 / 0"

    print("=== Dashboard Demo Start (ESC で終了) ===")
    while True:
        env.step(ctrl_left, ctrl_right)
        img_robot, img_global = env.get_images()
        state = env.get_robot_state(ctrl_left, ctrl_right)

        ui.render(
            img_robot=img_robot,
            img_global=img_global,
            robot_state=state,
            user_instruction=user_instruction,
            current_task=current_task,
            task_progress=task_progress,
            model_name=VISION_MODEL_NAME,
        )

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        time.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

