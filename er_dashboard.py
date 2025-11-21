import sys
import time
from dataclasses import dataclass
import json
import os
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
        QComboBox,
    )
except ImportError:
    print("PySide6 not found. Please install it.")
    sys.exit(1)


SCENE_PATH = "scene.xml"
# Robotics-ER model
VISION_MODEL_NAME = 'models/gemini-robotics-er-1.5-preview'

# Simulation speed adjustment
SIM_STEPS_PER_TICK = 40

@dataclass
class RobotState:
    position: np.ndarray  # (x, y, z)
    yaw_deg: float
    ctrl_left: float
    ctrl_right: float


class SimulationEnv:
    """MuJoCo Environment Wrapper"""

    def __init__(self, xml_path: str = SCENE_PATH):
        self.xml_path = xml_path
        self.model = None
        self.data = None
        self.robot_renderer = None
        self.global_renderer = None
        self.robot_bid = -1
        self.pole_joints = {}
        
        self.reload_scene(xml_path)

    def reload_scene(self, xml_path: str):
        if not os.path.exists(xml_path):
            print(f"Scene file not found: {xml_path}")
            return

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Renderers
        self.robot_renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.global_renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Robot body ID
        self.robot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")

        # Initialize pole joints for randomization
        self.pole_joints = self._init_pole_joints()
        
        # Initial randomization
        self.randomize_poles()

    def step(self, ctrl_left: float, ctrl_right: float):
        self.data.ctrl[0] = ctrl_right
        self.data.ctrl[1] = ctrl_left
        mujoco.mj_step(self.model, self.data)

    def get_images(self):
        # Robot View
        self.robot_renderer.update_scene(self.data, camera="robot_cam")
        img_robot = self.robot_renderer.render()
        # Global View
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
                adr_x = self.model.jnt_qposadr[jx]
                adr_y = self.model.jnt_qposadr[jy]
                pole_joints[color] = (adr_x, adr_y)
            except Exception:
                continue
        return pole_joints

    def randomize_poles(self):
        if not self.pole_joints:
            return
        # Randomize in front area
        for color, (adr_x, adr_y) in self.pole_joints.items():
            x = np.random.uniform(-2.0, 2.0)
            y = np.random.uniform(-4.0, -1.0)
            self.data.qpos[adr_x] = x
            self.data.qpos[adr_y] = y
            print(f"[POLE] {color} randomized to x={x:.2f}, y={y:.2f}")


class VisionSystem:
    """Wrapper for Robotics-ER Model"""

    def __init__(self, model_name: str = VISION_MODEL_NAME):
        self.model_name = model_name
        self.model = None

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️ GEMINI_API_KEY not found.")
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"✅ Vision model: {self.model_name}")
        except Exception as e:
            print(f"⚠️ Vision model init error: {e}")
            self.model = None

    def detect_object(self, img_rgb: np.ndarray, target_description: str):
        if self.model is None:
            return None

        from PIL import Image
        
        # Prompt from ai_robot_er.py
        prompt = f"""
        Detect the {target_description} in the image.
        Return the 2D bounding box in JSON format with keys "box_2d" containing [ymin, xmin, ymax, xmax].
        If no target is found, return null.
        Example output: {{"box_2d": [200, 300, 800, 400]}}
        """

        try:
            pil_img = Image.fromarray(img_rgb)
            response = self.model.generate_content(
                [prompt, pil_img],
                generation_config={"response_mime_type": "application/json"},
            )
            data = json.loads(response.text)
            
            if isinstance(data, dict) and "box_2d" in data and data["box_2d"]:
                box = data["box_2d"]
                print(f"[VISION] Found {target_description}: {box}")
                return box
            else:
                print(f"[VISION] {target_description} not found.")
                return None
        except Exception as e:
            print(f"⚠️ Vision detect error: {e}")
            return None


class RobotController:
    """Control logic from ai_robot_er.py"""

    def __init__(self):
        self.center_x = 500
        self.kp = 0.05
        self.base_speed = 15.0 # Forward speed

    def decide_action(self, bbox):
        # bbox: [ymin, xmin, ymax, xmax]
        if not bbox:
            # Stop if lost
            return [0.0, 0.0], False

        ymin, xmin, ymax, xmax = bbox
        obj_center_x = (xmin + xmax) / 2.0
        obj_height = ymax - ymin

        # Reached condition
        if obj_height > 900:
            print("🎯 Target Reached!")
            return [0.0, 0.0], True

        # P-Control
        error_x = obj_center_x - self.center_x
        turn = error_x * self.kp

        left = self.base_speed + turn
        right = self.base_speed - turn
        
        # Log for debugging
        print(f"🔍 [CTRL] cx={obj_center_x:.1f}, err={error_x:.1f}, turn={turn:.1f}, L/R={left:.1f}/{right:.1f}")

        return [np.clip(left, -20, 20), np.clip(right, -20, 20)], False


class VisionJob(QObject):
    """Async Vision Job"""
    finished = Signal()
    resultReady = Signal(object)

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robotics-ER Dashboard")

        # Components
        self.env = SimulationEnv()
        self.vision = VisionSystem()
        self.controller = RobotController()
        
        # State
        self.ctrl_left = 0.0
        self.ctrl_right = 0.0
        self.running = False
        self.vision_busy = False
        self.vision_thread = None
        self.vision_job = None
        self.bbox_display = None
        self.last_api_time = 0.0
        self.api_interval = 1.5 # 1.5s interval from ai_robot_er.py
        
        self.target_desc = "red vertical pole (cylinder target)"

        self._setup_ui()
        self._setup_timer()
        self._refresh_once()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout()
        central.setLayout(layout)

        # 1. Image Views
        self.label_global = QLabel("Global View")
        self.label_robot = QLabel("Robot View")
        for lbl in (self.label_global, self.label_robot):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setMinimumSize(320, 240)
            lbl.setStyleSheet("background-color: #202020; color: #AAAAAA;")
        
        layout.addWidget(self.label_global, 0, 0)
        layout.addWidget(self.label_robot, 0, 1)

        # 2. Info Panels
        self.label_info = QLabel()
        self.label_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label_info.setStyleSheet("color: white; background-color: #303030; padding: 5px;")
        layout.addWidget(self.label_info, 1, 0, 1, 2)

        # 3. Controls
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_randomize = QPushButton("Randomize Poles")
        self.combo_scene = QComboBox()
        self.combo_scene.addItem("scene.xml")
        self.combo_scene.addItem("scene_obstacle.xml") # Placeholder if user adds more
        
        self.edit_instruction = QLineEdit(self.target_desc)
        self.edit_instruction.setPlaceholderText("Target Description")

        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_randomize.clicked.connect(self.on_randomize)
        self.combo_scene.currentTextChanged.connect(self.on_scene_changed)
        self.edit_instruction.textChanged.connect(self.on_instruction_changed)

        control_layout.addWidget(QLabel("Scene:"))
        control_layout.addWidget(self.combo_scene)
        control_layout.addWidget(QLabel("Target:"))
        control_layout.addWidget(self.edit_instruction)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_randomize)

        layout.addWidget(control_panel, 2, 0, 1, 2)
        
        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 0)
        
        self.resize(1000, 700)

    def _setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(30)

    def _refresh_once(self):
        img_robot, img_global = self.env.get_images()
        self._update_image_label(self.label_global, img_global)
        self._update_image_label(self.label_robot, img_robot)
        self._update_info()

    def _on_tick(self):
        # Physics Step
        if self.running:
            for _ in range(SIM_STEPS_PER_TICK):
                self.env.step(self.ctrl_left, self.ctrl_right)

        # Rendering
        img_robot, img_global = self.env.get_images()
        
        # Vision Trigger
        if self.running and not self.vision_busy:
            now = time.time()
            if now - self.last_api_time > self.api_interval:
                self.last_api_time = now
                self._start_vision_job(img_robot, self.target_desc)

        # Overlay BBox
        robot_disp = img_robot.copy()
        if self.bbox_display:
            h, w, _ = robot_disp.shape
            ymin, xmin, ymax, xmax = self.bbox_display
            x1, y1 = int(xmin/1000*w), int(ymin/1000*h)
            x2, y2 = int(xmax/1000*w), int(ymax/1000*h)
            cv2.rectangle(robot_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Update UI
        self._update_image_label(self.label_global, img_global)
        self._update_image_label(self.label_robot, robot_disp)
        self._update_info()

    def _start_vision_job(self, img_rgb, target):
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

    def _on_vision_result(self, bbox):
        self.bbox_display = bbox
        new_ctrl, is_done = self.controller.decide_action(bbox)
        self.ctrl_left, self.ctrl_right = float(new_ctrl[0]), float(new_ctrl[1])
        
        if is_done:
            self.statusBar().showMessage("Target Reached!")
            self.running = False # Stop simulation/control on reach? Or just stop motors?
            self.ctrl_left = 0.0
            self.ctrl_right = 0.0

    def _on_vision_finished(self):
        self.vision_busy = False

    def _update_image_label(self, label, img_rgb):
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        if label.width() > 0 and label.height() > 0:
            pix = pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

    def _update_info(self):
        state = self.env.get_robot_state(self.ctrl_left, self.ctrl_right)
        text = (
            f"<b>Status:</b> {'RUNNING' if self.running else 'STOPPED'}<br>"
            f"<b>Target:</b> {self.target_desc}<br>"
            f"<b>Robot Pos:</b> ({state.position[0]:.2f}, {state.position[1]:.2f})<br>"
            f"<b>Yaw:</b> {state.yaw_deg:.1f} deg<br>"
            f"<b>Motors:</b> L={self.ctrl_left:.1f}, R={self.ctrl_right:.1f}<br>"
            f"<b>Vision Model:</b> {VISION_MODEL_NAME}"
        )
        self.label_info.setText(text)

    def on_start(self):
        self.running = True
        self.statusBar().showMessage("Simulation Started")

    def on_stop(self):
        self.running = False
        self.ctrl_left = 0.0
        self.ctrl_right = 0.0
        self.statusBar().showMessage("Simulation Stopped")

    def on_randomize(self):
        self.env.randomize_poles()
        self._refresh_once()
        self.statusBar().showMessage("Poles Randomized")

    def on_scene_changed(self, text):
        self.env.reload_scene(text)
        self._refresh_once()
        self.statusBar().showMessage(f"Scene loaded: {text}")

    def on_instruction_changed(self, text):
        self.target_desc = text

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
