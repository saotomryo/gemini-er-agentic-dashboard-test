import sys
import time
from dataclasses import dataclass
import json
import os
import csv
import math

import cv2
import mujoco
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

try:
    # Prefer PySide6
    from PySide6.QtCore import QTimer, Qt, QObject, QThread, Signal
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
        QFileDialog,
    )
except ImportError:  # fallback to PySide2
    from PySide2.QtCore import QTimer, Qt, QObject, QThread, Signal
    from PySide2.QtGui import QImage, QPixmap
    from PySide2.QtWidgets import (
        QApplication,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
        QFileDialog,
    )


SCENE_PATH = "scene.xml"
VISION_MODEL_NAME = "gemini-2.5-flash"  # 将来 ER (gemini-robotics-er) へ差し替え予定
PLANNER_MODEL_NAME = "gemini-2.5-flash"

# シミュレーション速度調整用（1ティックで何ステップ進めるか）
# 回転・前進が十分進むように少し大きめにする
SIM_STEPS_PER_TICK = 40

# move_to 完了後に安全距離を取るためのバック走行時間（タイマー tick 単位）
BACKUP_TICKS_AFTER_MOVE = 30


@dataclass
class RobotState:
    position: np.ndarray  # (x, y, z)
    yaw_deg: float
    ctrl_left: float
    ctrl_right: float


class SimulationEnv:
    """MuJoCo 環境のラッパ（PySide 用）"""

    def __init__(self, xml_path: str = SCENE_PATH):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # レンダラー
        self.robot_renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.global_renderer = mujoco.Renderer(self.model, height=480, width=640)

        # ロボット body ID
        self.robot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")

        # ポール用ジョイント情報（ランダム配置用）
        self.pole_joints = self._init_pole_joints()

        # 起動時もランダム配置にする
        self.randomize_poles()

    def step(self, ctrl_left: float, ctrl_right: float):
        # actuator 並び: motor_right, motor_left
        self.data.ctrl[0] = ctrl_right
        self.data.ctrl[1] = ctrl_left
        mujoco.mj_step(self.model, self.data)

    def get_images(self):
        # ロボット視点
        self.robot_renderer.update_scene(self.data, camera="robot_cam")
        img_robot = self.robot_renderer.render()
        # 俯瞰視点
        self.global_renderer.update_scene(self.data, camera="global_cam")
        img_global = self.global_renderer.render()
        return img_robot, img_global

    def get_robot_state(self, ctrl_left: float, ctrl_right: float) -> RobotState:
        pos = np.array(self.data.xpos[self.robot_bid])
        rot = np.array(self.data.xmat[self.robot_bid]).reshape(3, 3)
        forward = rot[:, 0]
        yaw = np.degrees(np.arctan2(forward[1], forward[0]))
        return RobotState(position=pos, yaw_deg=yaw, ctrl_left=ctrl_left, ctrl_right=ctrl_right)

    def _init_pole_joints(self):
        """scene.xml に定義した各色ポールの slide ジョイントを登録"""
        pole_defs = [
            ("red", "joint_target_red_x", "joint_target_red_y"),
            ("blue", "joint_target_blue_x", "joint_target_blue_y"),
            ("yellow", "joint_target_yellow_x", "joint_target_yellow_y"),
        ]
        pole_joints = {}
        for color, jx_name, jy_name in pole_defs:
            try:
                jx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jx_name)
                jy = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jy_name)
            except Exception:
                continue
            adr_x = self.model.jnt_qposadr[jx]
            adr_y = self.model.jnt_qposadr[jy]
            pole_joints[color] = (adr_x, adr_y)
        return pole_joints

    def randomize_poles(self):
        """ポールの位置をランダムに配置し直す"""
        if not self.pole_joints:
            return

        # ロボットの前方エリアにランダム配置 (x: [-2, 2], y: [-4, -1] など)
        for color, (adr_x, adr_y) in self.pole_joints.items():
            x = np.random.uniform(-2.0, 2.0)
            y = np.random.uniform(-4.0, -1.0)
            self.data.qpos[adr_x] = x
            self.data.qpos[adr_y] = y
            print(f"[POLE] {color} at x={x:.2f}, y={y:.2f}")

    def get_pole_positions(self):
        """現在の各ポールの (x, y) 位置を取得"""
        positions = {}
        for color, (adr_x, adr_y) in self.pole_joints.items():
            x = float(self.data.qpos[adr_x])
            y = float(self.data.qpos[adr_y])
            positions[color] = (x, y)
        return positions

    def reset_poles_to_default(self):
        """ポールの位置を固定の初期位置に設定する"""
        if not self.pole_joints:
            return

        # 固定初期位置 (ロボット前方に3本並べる例)
        defaults = {
            "red": (0.0, -3.0),
            "blue": (2.0, -3.0),
            "yellow": (-2.0, -3.0),
        }
        for color, (adr_x, adr_y) in self.pole_joints.items():
            x, y = defaults.get(color, (0.0, -3.0))
            self.data.qpos[adr_x] = x
            self.data.qpos[adr_y] = y
            print(f"[POLE] {color} default at x={x:.2f}, y={y:.2f}")



class VisionSystem:
    """Gemini を用いた簡易物体検出"""

    def __init__(self, model_name: str = VISION_MODEL_NAME):
        self.model_name = model_name
        self.model = None

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️ GEMINI_API_KEY が設定されていません。Vision は無効です。")
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"✅ Vision model: {self.model_name}")
        except Exception as e:
            print(f"⚠️ Vision モデル初期化エラー: {e}")
            self.model = None

    def detect_object(self, img_rgb: np.ndarray, target_description: str):
        """画像とターゲット説明から 2D バウンディングボックスと到達判定を取得

        戻り値:
            (box_2d, reached)
            box_2d: [ymin, xmin, ymax, xmax] (0-1000 正規化) または None
            reached: bool または None
        """
        if self.model is None:
            return None, None

        from PIL import Image

        prompt = f"""
        You are controlling a mobile robot with an onboard camera.
        Find the "{target_description}" in the image.

        1. Detect the 2D bounding box of the target object as [ymin, xmin, ymax, xmax] in 0-1000 normalized coordinates.
        2. Decide whether the robot has already "reached" the target:
           - reached = true: the robot is close enough that it should stop and not move closer
           - reached = false: the robot is still far enough away that it should keep approaching

        Output JSON object:
        {{
          "box_2d": [ymin, xmin, ymax, xmax] | null,
          "reached": true | false
        }}

        If the target is not visible, set "box_2d" to null and "reached" to false.
        """

        try:
            pil_img = Image.fromarray(img_rgb)
            response = self.model.generate_content(
                [prompt, pil_img],
                generation_config={"response_mime_type": "application/json"},
            )
            data = json.loads(response.text)
            box = None
            reached = None
            if isinstance(data, dict):
                box = data.get("box_2d")
                reached = data.get("reached")

            if box:
                print(f"[VISION] target='{target_description}' box_2d={box} reached={reached}")
            else:
                print(f"[VISION] target='{target_description}' not found (reached={reached})")
            return box, reached
        except Exception as e:
            print(f"⚠️ Vision detect error: {e}")
        return None, None


class RobotController:
    """
    画像上のターゲット位置だけを見て制御するコントローラ。
    general_agent_v4.py とほぼ同じロジックに戻し、
    あたり判定のみ少し厳しめに調整しています。
    """

    def __init__(self):
        self.center_x = 500        # 画面中央 (0-1000)
        self.align_threshold = 80  # 中央とみなす範囲(±80)

        # プラスが前進方向（general_agent_v4.py と揃える）
        self.base_speed = 15.0     # 直進時の基本速度
        self.turn_speed = 8.0      # 旋回時の速度
        self.kp = 0.02             # 旋回微調整のゲイン

        # 到達判定用（連続フレーム）
        self.close_frames = 0

    def decide_action(self, bbox, task_action: str):
        # ターゲットが見つからない -> 探索用の旋回
        if not bbox:
            print(f"[CTRL] action={task_action} bbox=None -> SEARCH (spin)")
            self.close_frames = 0
            return [self.turn_speed, -self.turn_speed], False

        ymin, xmin, ymax, xmax = bbox
        obj_center_x = (xmin + xmax) / 2.0
        obj_height = ymax - ymin

        # 画面中央との誤差 (右にあるとプラス)
        error = obj_center_x - self.center_x

        print(
            f"[CTRL] action={task_action} cx={obj_center_x:.1f} "
            f"err={error:.1f} h={obj_height:.1f}"
        )

        # --- look_at: 方向だけ合わせる ---
        if task_action == "look_at":
            if abs(error) < self.align_threshold:
                print("[CTRL] look_at -> DONE (centered)")
                return [0.0, 0.0], True

            print("[CTRL] look_at -> ALIGN")
            # ログから実挙動を踏まえて符号を調整:
            # error > 0 (ターゲットが右) なら左回転、error < 0 (左) なら右回転
            if error > 0:
                # 右にある -> 左旋回
                return [-self.turn_speed, self.turn_speed], False
            else:
                # 左にある -> 右旋回
                return [self.turn_speed, -self.turn_speed], False

        # --- move_to: 方向を合わせてから前進 ---
        if task_action == "move_to":
            # あたり判定: 画像上でかなり大きく写っている状態が数フレーム続いたら到達
            if obj_height > 900:
                self.close_frames += 1
                if self.close_frames >= 3:
                    print("[CTRL] move_to -> DONE (close enough, stable)")
                    self.close_frames = 0
                    return [0.0, 0.0], True
                else:
                    print(f"[CTRL] move_to -> CLOSE ({self.close_frames}/3)")
            else:
                self.close_frames = 0

            # まずは向き調整
            if abs(error) > self.align_threshold:
                print("[CTRL] move_to -> ALIGN (turn)")
                # look_at と同じく、error>0: 左回転 / error<0: 右回転
                if error > 0:
                    return [-self.turn_speed, self.turn_speed], False
                else:
                    return [self.turn_speed, -self.turn_speed], False

            # だいたい正面に入ったら前進しながら微調整
            print("[CTRL] move_to -> APPROACH (forward)")
            correction = error * self.kp
            left = self.base_speed + correction
            right = self.base_speed - correction
            return [left, right], False

        return [0.0, 0.0], False


class TaskPlanner:
    """自然言語の指示をタスクリストに変換する Planner"""

    def __init__(self, model_name: str = PLANNER_MODEL_NAME):
        self.model_name = model_name
        self.model = None

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️ GEMINI_API_KEY が設定されていません。Planner は無効です。")
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"✅ Planner model: {self.model_name}")
        except Exception as e:
            print(f"⚠️ Planner モデル初期化エラー: {e}")
            self.model = None

    def plan_tasks(self, instruction: str):
        """指示文を [{action, target}, ...] の JSON 配列に変換"""
        if self.model is None:
            return []

        prompt = f"""
        You control a simple mobile robot in simulation.
        Convert the following natural language command into a sequence of low-level tasks.

        Command: "{instruction}"

        Available actions:
        - "move_to": move close to the specified object.
        - "look_at": turn to face the specified object without approaching closely.

        Output format: JSON array, each item:
        {{
          "action": "move_to" | "look_at",
          "target": "short English description of object (e.g. 'red pole', 'blue pole', 'yellow pole')"
        }}

        Return ONLY the JSON array. If you cannot interpret the command into these actions, return an empty JSON array [].
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            data = json.loads(response.text)
            if isinstance(data, list):
                return data
        except Exception as e:
            print(f"⚠️ plan_tasks error: {e}")
        return []


class VisionJob(QObject):
    """Vision 推論を UI スレッドとは別スレッドで実行するジョブ"""

    finished = Signal()
    resultReady = Signal(object)  # bbox または None

    def __init__(self, vision: "VisionSystem", img_rgb: np.ndarray, target: str):
        super().__init__()
        self._vision = vision
        self._img_rgb = img_rgb
        self._target = target

    def run(self):
        result = self._vision.detect_object(self._img_rgb, self._target)
        self.resultReady.emit(result)
        self.finished.emit()


class MainWindow(QMainWindow):
    """PySide ベースのダッシュボード UI"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Dashboard (PySide)")

        # シミュレータ
        self.env = SimulationEnv()
        self.ctrl_left = 0.0
        self.ctrl_right = 0.0
        self.running = False  # Start/Stop ボタンで制御

        # Vision / Controller
        self.vision = VisionSystem()
        self.controller = RobotController()
        self.vision_busy = False
        self.vision_thread = None
        self.vision_job = None

        # move_to 完了後のバック走行管理
        self.backup_ticks_remaining = 0

        # チューニング用ログの蓄積
        self.tune_records = []

        # Planner
        self.planner = TaskPlanner()

        # タスク（まずは赤いポールに近づく 1 ステップのみ）
        self.tasks = [{"action": "move_to", "target": "red pole"}]
        self.current_task_idx = 0
        self.bbox_display = None
        self.last_api_time = 0.0
        # Gemini Vision 呼び出し間隔（秒）: 制御追従性を優先して短めに
        self.api_interval = 1.0  # [s]

        # ユーザー指示 / タスク情報（とりあえず固定テキスト）
        self.user_instruction = "赤いポールに近づいてください。"
        self.current_task = "move_to red pole"
        self.task_progress = "1 / 1"

        self._setup_ui()
        self.statusBar().showMessage("Ready: set instruction and press Start")
        # 起動直後に一度描画して、カメラ画像と情報を表示しておく
        self._refresh_once()
        self._setup_timer()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QGridLayout()
        central.setLayout(layout)

        # 上段: 画像ビュー
        self.label_global = QLabel("Global View")
        self.label_robot = QLabel("Robot View")

        for lbl in (self.label_global, self.label_robot):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setMinimumSize(320, 240)
            lbl.setStyleSheet("background-color: #202020; color: #AAAAAA;")

        layout.addWidget(self.label_global, 0, 0)
        layout.addWidget(self.label_robot, 0, 1)

        # 下段左: テキスト指示 / タスク
        self.label_info_left = QLabel()
        self.label_info_left.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label_info_left.setStyleSheet("color: white;")
        self.label_info_left.setWordWrap(True)

        # 下段右: 状態 / モデル情報
        self.label_info_right = QLabel()
        self.label_info_right.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label_info_right.setStyleSheet("color: white;")
        self.label_info_right.setWordWrap(True)

        info_left_container = QWidget()
        info_left_layout = QVBoxLayout()
        info_left_layout.addWidget(self.label_info_left)
        info_left_container.setLayout(info_left_layout)
        info_left_container.setStyleSheet("background-color: #303030;")

        info_right_container = QWidget()
        info_right_layout = QVBoxLayout()
        info_right_layout.addWidget(self.label_info_right)
        info_right_container.setLayout(info_right_layout)
        info_right_container.setStyleSheet("background-color: #303030;")

        layout.addWidget(info_left_container, 1, 0)
        layout.addWidget(info_right_container, 1, 1)

        # プロンプト入力行
        prompt_container = QWidget()
        prompt_layout = QHBoxLayout()
        prompt_container.setLayout(prompt_layout)

        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("例: 赤いポールに近づいて、そのあと青いポールの方向を向いて")
        self.btn_apply_prompt = QPushButton("Apply Prompt")
        self.btn_apply_prompt.clicked.connect(self.on_apply_prompt_clicked)

        prompt_layout.addWidget(QLabel("Instruction:"))
        prompt_layout.addWidget(self.prompt_edit, 1)
        prompt_layout.addWidget(self.btn_apply_prompt)

        layout.addWidget(prompt_container, 2, 0, 1, 2)

        # コントロールボタン行
        button_container = QWidget()
        button_layout = QHBoxLayout()
        button_container.setLayout(button_layout)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_randomize = QPushButton("Randomize Poles")
        self.btn_export_log = QPushButton("Export Log")
        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        self.btn_randomize.clicked.connect(self.on_randomize_poles)
        self.btn_export_log.clicked.connect(self.on_export_log_clicked)

        button_layout.addStretch(1)
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_stop)
        button_layout.addWidget(self.btn_randomize)
        button_layout.addWidget(self.btn_export_log)
        button_layout.addStretch(1)

        layout.addWidget(button_container, 3, 0, 1, 2)

        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 0)
        layout.setRowStretch(3, 0)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        self.resize(1200, 800)

    def _setup_timer(self):
        # QTimer でシミュレーションと描画を回す
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(30)  # 約 30ms ごと (~33fps)

    def _refresh_once(self):
        """起動直後用の1回だけの描画処理"""
        img_robot, img_global = self.env.get_images()
        state = self.env.get_robot_state(self.ctrl_left, self.ctrl_right)
        self._update_image_label(self.label_global, img_global)
        self._update_image_label(self.label_robot, img_robot)
        self._update_info_labels(state)

    def _start_vision_job(self, img_rgb: np.ndarray, target: str):
        """Gemini Vision を別スレッドで実行"""
        if self.vision is None or self.vision.model is None:
            return

        print(f"[VISION] enqueue job target='{target}'")
        # 推論中は、探索や軸合わせ中のみ一旦モーターを止める。
        # すでに正面を向いて前進しているときは止めずに走り続ける。
        stop_for_vision = True
        if (
            self.current_task_idx < len(self.tasks)
            and isinstance(self.bbox_display, (list, tuple))
            and len(self.bbox_display) >= 4
        ):
            ymin, xmin, ymax, xmax = self.bbox_display[:4]
            cx = (xmin + xmax) / 2.0
            error = cx - self.controller.center_x
            task = self.tasks[self.current_task_idx]
            if task["action"] == "move_to" and abs(error) <= self.controller.align_threshold:
                # ほぼ正面を向いて move_to 中 → 直進状態なので止めない
                stop_for_vision = False

        if stop_for_vision:
            self.ctrl_left = 0.0
            self.ctrl_right = 0.0
        self.vision_busy = True

        self.vision_thread = QThread(self)
        self.vision_job = VisionJob(self.vision, img_rgb.copy(), target)
        self.vision_job.moveToThread(self.vision_thread)

        self.vision_thread.started.connect(self.vision_job.run)
        self.vision_job.resultReady.connect(self._on_vision_result)
        self.vision_job.finished.connect(self._on_vision_finished)
        self.vision_job.finished.connect(self.vision_thread.quit)
        self.vision_job.finished.connect(self.vision_job.deleteLater)
        self.vision_thread.finished.connect(self.vision_thread.deleteLater)

        self.vision_thread.start()

    def _on_vision_result(self, result):
        """Vision 推論結果を受け取って制御に反映"""
        if self.current_task_idx >= len(self.tasks):
            return

        # VisionSystem からの戻り値は (bbox, reached) を想定
        reached = None
        if isinstance(result, tuple) and len(result) == 2:
            bbox, reached = result
        else:
            bbox = result

        task = self.tasks[self.current_task_idx]
        self.bbox_display = bbox

        new_ctrl, is_done = self.controller.decide_action(bbox, task["action"])
        new_ctrl = np.clip(new_ctrl, -20, 20)
        self.ctrl_left, self.ctrl_right = float(new_ctrl[0]), float(new_ctrl[1])

        print(
            f"[STEP] ctrl L/R = {self.ctrl_left:+.1f}, {self.ctrl_right:+.1f} "
            f"(task={task['action']}:{task['target']})"
        )

        # チューニング用の詳細ログ
        self._log_motion(task, bbox, self.ctrl_left, self.ctrl_right)

        # Vision 側の reached フラグは参考情報とし、
        # 実際の到達判定はコントローラ側の高さしきい値に委ねる

        if is_done:
            # move_to 完了後は少しバックしてから次のタスクへ進む
            if task["action"] == "move_to":
                self.backup_ticks_remaining = BACKUP_TICKS_AFTER_MOVE
                back_speed = -self.controller.base_speed * 0.5
                self.ctrl_left = back_speed
                self.ctrl_right = back_speed
                self.statusBar().showMessage(
                    "Backing up after move_to, then next task"
                )
            else:
                # look_at などはその場で停止
                self.ctrl_left = 0.0
                self.ctrl_right = 0.0

            self.current_task_idx += 1

            if self.current_task_idx >= len(self.tasks):
                # 全タスク完了。バック走行中であっても、完了後に停止する。
                if self.backup_ticks_remaining == 0:
                    self.ctrl_left = 0.0
                    self.ctrl_right = 0.0
                self.statusBar().showMessage("Done: all tasks completed")
            else:
                next_task = self.tasks[self.current_task_idx]
                # すぐに次タスクに入る場合のみステータスを更新。
                # バック走行中は、バック完了時に _on_tick 側で改めてステータスを更新する。
                if self.backup_ticks_remaining == 0:
                    self.statusBar().showMessage(
                        f"Running: next task {self.current_task_idx + 1}/{len(self.tasks)} "
                        f"{next_task['action']} -> {next_task['target']}"
                    )

    def _on_vision_finished(self):
        self.vision_busy = False

    def on_start_clicked(self):
        self.running = True
        if self.tasks and self.current_task_idx < len(self.tasks):
            task = self.tasks[self.current_task_idx]
            self.statusBar().showMessage(
                f"Running: task {self.current_task_idx + 1}/{len(self.tasks)} "
                f"{task['action']} -> {task['target']}"
            )
        else:
            self.statusBar().showMessage("Warning: no valid tasks (NG)")

    def on_stop_clicked(self):
        self.running = False
        self.statusBar().showMessage("Paused")

    def on_randomize_poles(self):
        # シミュレーションを一瞬止めてから配置変更
        was_running = self.running
        self.running = False
        self.env.randomize_poles()
        # 位置変更を画面にすぐ反映させるため、少しだけステップを進める
        for _ in range(10):
            self.env.step(self.ctrl_left, self.ctrl_right)
        self.running = was_running
        self.statusBar().showMessage("Poles randomized")

    def on_apply_prompt_clicked(self):
        """UI からの自然言語指示を Planner にかけてタスクリストを更新"""
        text = self.prompt_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "NG", "指示文が空です。")
            self._set_ng_state(text, reason="empty")
            self.statusBar().showMessage("NG: empty instruction")
            return

        tasks = self.planner.plan_tasks(text)
        valid_tasks = self._validate_tasks(tasks)
        if not valid_tasks:
            QMessageBox.warning(self, "NG", "指示を有効なタスクに変換できませんでした。")
            self._set_ng_state(text, reason="invalid_tasks")
            self.statusBar().showMessage("NG: could not interpret instruction")
            return

        # 正常にタスクリストを更新
        self.user_instruction = text
        self.tasks = valid_tasks
        self.current_task_idx = 0
        self.bbox_display = None
        self.ctrl_left = 0.0
        self.ctrl_right = 0.0
        self.last_api_time = 0.0
        self.backup_ticks_remaining = 0

        # 進捗表示を更新
        if self.tasks:
            t0 = self.tasks[0]
            self.current_task = f"{t0['action']} -> {t0['target']}"
            self.task_progress = f"1 / {len(self.tasks)}"

        print(f"[PLANNER] tasks = {self.tasks}")
        self.statusBar().showMessage("OK: tasks planned, press Start")

    def _validate_tasks(self, tasks):
        """LLM が返したタスクリストが扱える形かどうかをチェック"""
        if not isinstance(tasks, list) or not tasks:
            return []

        allowed_actions = {"move_to", "look_at"}
        cleaned = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            action = t.get("action")
            target = t.get("target")
            if action not in allowed_actions:
                continue
            if not isinstance(target, str) or not target:
                continue
            cleaned.append({"action": action, "target": target})

        return cleaned

    def _set_ng_state(self, instruction_text: str, reason: str = ""):
        """解釈 NG のときの状態更新"""
        self.user_instruction = instruction_text + (" (NG)" if instruction_text else "NG")
        self.tasks = []
        self.current_task_idx = 0
        self.current_task = "NG"
        self.task_progress = "NG"
        self.ctrl_left = 0.0
        self.ctrl_right = 0.0
        self.bbox_display = None
        self.backup_ticks_remaining = 0
        print(f"[PLANNER] NG ({reason}) for instruction: {instruction_text!r}")

    def _on_tick(self):
        # 1. MuJoCo ステップ（Start 中のみ）
        if self.running:
            # 1回のタイマーで複数ステップ進めて、動きをわかりやすくする
            for _ in range(SIM_STEPS_PER_TICK):
                self.env.step(self.ctrl_left, self.ctrl_right)

            # move_to 完了後のバック走行時間をカウントダウン
            if self.backup_ticks_remaining > 0:
                self.backup_ticks_remaining -= 1
                # バック走行が終了したら一旦停止し、次のタスクへ
                if self.backup_ticks_remaining == 0:
                    self.ctrl_left = 0.0
                    self.ctrl_right = 0.0
                    if self.current_task_idx < len(self.tasks):
                        task = self.tasks[self.current_task_idx]
                        self.statusBar().showMessage(
                            f"Running: task {self.current_task_idx + 1}/{len(self.tasks)} "
                            f"{task['action']} -> {task['target']}"
                        )

        # 2. 画像と状態を取得
        img_robot, img_global = self.env.get_images()
        state = self.env.get_robot_state(self.ctrl_left, self.ctrl_right)

        # 3. Vision / 制御（一定間隔でジョブを投げる・結果は非同期コールバック）
        # バック走行中は Vision を呼ばない
        if (
            self.running
            and self.current_task_idx < len(self.tasks)
            and not self.vision_busy
            and self.backup_ticks_remaining == 0
        ):
            now = time.time()
            if now - self.last_api_time > self.api_interval:
                task = self.tasks[self.current_task_idx]
                self.last_api_time = now
                self._start_vision_job(img_robot, task["target"])

        # 4. ロボット画像に検出結果をオーバーレイ
        robot_disp = img_robot.copy()
        if isinstance(self.bbox_display, (list, tuple)) and len(self.bbox_display) >= 4:
            h, w, _ = robot_disp.shape
            ymin, xmin, ymax, xmax = self.bbox_display[:4]
            x1 = int(xmin / 1000 * w)
            x2 = int(xmax / 1000 * w)
            y1 = int(ymin / 1000 * h)
            y2 = int(ymax / 1000 * h)
            cv2.rectangle(robot_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 画面中央ガイド
            cv2.line(robot_disp, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)

        # タスク情報を更新
        if self.current_task_idx < len(self.tasks):
            task = self.tasks[self.current_task_idx]
            self.current_task = f"{task['action']} -> {task['target']}"
            self.task_progress = f"{self.current_task_idx + 1} / {len(self.tasks)}"
        else:
            self.current_task = "All tasks completed"
            self.task_progress = f"{len(self.tasks)} / {len(self.tasks)}"

        # 5. ラベルに画像を反映
        self._update_image_label(self.label_global, img_global)
        self._update_image_label(self.label_robot, robot_disp)

        # 6. テキスト情報更新
        self._update_info_labels(state)

    @staticmethod
    def _numpy_rgb_to_qimage(img: np.ndarray) -> QImage:
        h, w, ch = img.shape
        assert ch == 3
        bytes_per_line = ch * w
        # OpenGL → numpy は行が連続している前提
        return QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    def _update_image_label(self, label: QLabel, img_rgb: np.ndarray):
        qimg = self._numpy_rgb_to_qimage(img_rgb)
        pix = QPixmap.fromImage(qimg)
        # ラベルサイズにフィット（まだサイズが決まっていない起動直後は元サイズのまま表示）
        target_size = label.size()
        if target_size.width() > 0 and target_size.height() > 0:
            pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

    def _update_info_labels(self, state: RobotState):
        # 左側: 指示・タスク
        left_text = (
            f"<b>Instruction</b><br>{self.user_instruction}<br><br>"
            f"<b>Current Task</b><br>{self.current_task}<br><br>"
            f"<b>Progress</b><br>{self.task_progress}<br>"
        )
        self.label_info_left.setText(left_text)

        # 右側: ロボット状態 / モデル
        pos = state.position
        # ポール位置情報
        pole_pos = self.env.get_pole_positions()
        pole_lines = []
        for color, (px, py) in sorted(pole_pos.items()):
            pole_lines.append(f"{color}: ({px:+.2f}, {py:+.2f})")
        poles_html = "<br>".join(pole_lines) if pole_lines else "N/A"

        right_text = (
            "<b>Robot State</b><br>"
            f"pos: ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})<br>"
            f"yaw: {state.yaw_deg:+.1f} deg<br>"
            f"ctrl L/R: {state.ctrl_left:+.1f}, {state.ctrl_right:+.1f}<br><br>"
            "<b>Poles (x,y)</b><br>"
            f"{poles_html}<br><br>"
            "<b>Models</b><br>"
            f"Vision: {VISION_MODEL_NAME}<br>"
            f"Planner: {PLANNER_MODEL_NAME}<br>"
        )
        self.label_info_right.setText(right_text)

    def _log_motion(self, task, bbox, ctrl_left: float, ctrl_right: float):
        """対象物・移動方向・位置などをまとめてログ出力し、調整に使えるようにする"""
        action = task.get("action")
        target = task.get("target")

        # 移動方向の分類
        move_mode = "stop"
        if abs(ctrl_left) < 0.1 and abs(ctrl_right) < 0.1:
            move_mode = "stop"
        elif ctrl_left * ctrl_right > 0:
            # general_agent_v4 と同じく「プラスが前進方向」
            move_mode = "forward" if ctrl_left > 0 else "backward"
        else:
            # 差分で左右旋回を判定
            if ctrl_left > ctrl_right:
                move_mode = "turn_right"
            elif ctrl_left < ctrl_right:
                move_mode = "turn_left"

        # 画像上でのターゲット位置のざっくり分類
        side = "none"
        center_norm = None
        height = None
        if bbox:
            ymin, xmin, ymax, xmax = bbox
            cx = (xmin + xmax) / 2.0  # 0-1000
            center_norm = cx / 1000.0
            height = ymax - ymin
            if cx < self.controller.center_x - self.controller.align_threshold:
                side = "left"
            elif cx > self.controller.center_x + self.controller.align_threshold:
                side = "right"
            else:
                side = "center"

        # 現在のワールド座標・ヨー角
        state = self.env.get_robot_state(ctrl_left, ctrl_right)
        px, py, pz = state.position

        # 対象ポールとの距離・方位差を計算
        target_distance = None
        heading_diff_deg = None
        pole_pos = self.env.get_pole_positions()
        target_color = None
        if isinstance(target, str):
            t_lower = target.lower()
            for c in ("red", "blue", "yellow"):
                if c in t_lower:
                    target_color = c
                    break

        if target_color and target_color in pole_pos:
            tx, ty = pole_pos[target_color]
            dx = tx - px
            dy = ty - py
            target_distance = math.sqrt(dx * dx + dy * dy)
            target_bearing = math.degrees(math.atan2(dy, dx))
            # ロボットの向きとの差分 [-180, 180] に正規化
            raw_diff = target_bearing - state.yaw_deg
            heading_diff_deg = ((raw_diff + 180.0) % 360.0) - 180.0

        # 数値/NA を文字列に整形
        if center_norm is None:
            center_str = "NA"
        else:
            center_str = f"{center_norm:.3f}"

        if height is None:
            height_str = "NA"
        else:
            height_str = f"{height:.1f}"

        if target_distance is None:
            dist_str = "NA"
        else:
            dist_str = f"{target_distance:.2f}"

        if heading_diff_deg is None:
            heading_str = "NA"
        else:
            heading_str = f"{heading_diff_deg:+.1f}"

        # 記録用に辞書として保持
        record = {
            "time": time.time(),
            "action": action,
            "target": target,
            "move_mode": move_mode,
            "bbox_side": side,
            "bbox_center": center_norm,
            "bbox_height": height,
            "target_distance": target_distance,
            "heading_diff_deg": heading_diff_deg,
            "pos_x": px,
            "pos_y": py,
            "pos_z": pz,
            "yaw_deg": state.yaw_deg,
            "ctrl_left": ctrl_left,
            "ctrl_right": ctrl_right,
        }
        self.tune_records.append(record)

        print(
            "[TUNE] "
            f"task={action}:{target} "
            f"move={move_mode} "
            f"bbox_side={side} "
            f"bbox_center={center_str} "
            f"bbox_height={height_str} "
            f"dist={dist_str} "
            f"heading_diff={heading_str}deg "
            f"pos=({px:+.2f},{py:+.2f},{pz:+.2f}) "
            f"yaw={state.yaw_deg:+.1f}deg "
            f"ctrl=({ctrl_left:+.1f},{ctrl_right:+.1f})"
        )

    def on_export_log_clicked(self):
        """チューニング用ログをファイルに書き出す"""
        if not self.tune_records:
            QMessageBox.information(self, "Export Log", "まだログがありません。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export tuning log",
            "tuning_log.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        fieldnames = [
            "time",
            "action",
            "target",
            "move_mode",
            "bbox_side",
            "bbox_center",
            "bbox_height",
            "target_distance",
            "heading_diff_deg",
            "pos_x",
            "pos_y",
            "pos_z",
            "yaw_deg",
            "ctrl_left",
            "ctrl_right",
        ]
        try:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.tune_records:
                    writer.writerow(r)
            self.statusBar().showMessage(f"Log exported to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Export Log", f"書き出しに失敗しました: {e}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    print("PySide Dashboard running. Close the window to exit.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
