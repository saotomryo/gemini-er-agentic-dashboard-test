import sys
import time
from dataclasses import dataclass
import json
import os
import math
from pathlib import Path
from typing import List, Optional

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
        QListWidget
    )
except ImportError:
    print("PySide6 not found. Please install it.")
    sys.exit(1)


SCENE_PATH = "scene.xml"
# Robotics-ER model for Vision
VISION_MODEL_NAME = "models/gemini-robotics-er-1.5-preview"
PLANNER_MODEL_NAME = "models/gemini-robotics-er-1.5-preview"

# Simulation Constants
SIM_STEPS_PER_TICK = 5
SCAN_TURN_SPEED = 3.5   # Speed used during scan rotations
SCAN_TURN_DURATION = 4.0  # Seconds to approximate 90deg turn
DEFAULT_TURN_SPEED = 6.0   # Fallback discrete turn speed (calibration can override)
DEFAULT_TURN_DURATION = 2.0  # Fallback discrete turn duration
CALIBRATION_FILE = Path("calibration_turn.json")


def load_turn_calibration():
    if not CALIBRATION_FILE.exists():
        return None
    try:
        with CALIBRATION_FILE.open("r") as f:
            data = json.load(f)
        speed = float(data.get("turn_speed", DEFAULT_TURN_SPEED))
        duration = float(data.get("turn_duration", DEFAULT_TURN_DURATION))
        return {"speed": speed, "duration": duration}
    except Exception as e:
        print(f"⚠️ Failed to read {CALIBRATION_FILE}: {e}")
        return None

# Debug toggle
DEBUG_MODE = os.getenv("ER_DEBUG", "0") == "1"

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


class AgentPlanner:
    """Planner using Gemini Flash"""
    def __init__(self, model_name: str = None):
        # Use the global PLANNER_MODEL_NAME if no model_name provided
        if model_name is None:
            model_name = PLANNER_MODEL_NAME
        self.model_name = model_name
        self.model = None
        
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"✅ Planner model: {self.model_name}")

    def decide_next_step(self, instruction: str, history: list, img_rgb: np.ndarray, surround_imgs: list = None):
        if not self.model:
            return None
        
        from PIL import Image
        
        history_str = "\n".join([f"- {h}" for h in history[-5:]]) # Keep last 5 actions
        
        prompt_text = f"""
        You are a robot agent operating in a simulation.
        Your goal is to execute the user's instruction: "{instruction}"
        
        History of recent actions:
        {history_str}
        
        Based on the current view (image) and history, decide the NEXT IMMEDIATE discrete action.
        
        Available Actions:
        - "scan": Rotate 360 degrees to capture surround view. Use this FIRST if you haven't scanned yet or need to find a target.
        - "turn_left_30": Turn left by approx 30 degrees. Use this to search or adjust heading.
        - "turn_right_30": Turn right by approx 30 degrees.
        - "move_forward_short": Move forward for a short distance (approx 0.5s).
        - "approach": Use visual servoing to move towards a visible target. Requires "target" param.
        - "stop": Stop if the instruction is fully completed.
        
        Rules:
        1. If you have NO surround images (first step), use "scan".
        2. If the target is NOT visible in the current view, check if it was seen in the surround scan. If so, turn towards it.
        3. If the target IS visible, use "approach" with the target name.
        4. Do NOT generate a list of tasks. Just ONE action.
        
        Output JSON object (no prose):
        {{"action": "<action_name>", "target": "<target_if_needed>", "reason": "<brief>"}}
        """
        
        contents = [prompt_text]
        if img_rgb is not None:
            pil_img = Image.fromarray(img_rgb)
            contents.append("Current View:")
            contents.append(pil_img)
            
        if surround_imgs:
            contents.append("Surround View (Front, Right, Back, Left):")
            for i, img in enumerate(surround_imgs):
                contents.append(Image.fromarray(img))
        
        try:
            response = self.model.generate_content(contents, generation_config={"response_mime_type": "application/json"})
            step = json.loads(response.text)
            return step
        except Exception as e:
            print(f"⚠️ Planning error: {e}")
            return None


class ERPlannerAssist:
    """Fallback planner using ER directly for a single-step JSON action."""
    def __init__(self, model_name: str = VISION_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.last_prompt = None
        self.last_response = None
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            try:
                self.model = genai.GenerativeModel(self.model_name)
                print(f"✅ ER fallback planner: {self.model_name}")
            except Exception as e:
                print(f"⚠️ ER fallback init error: {e}")

    def plan(self, instruction: str, history: list, img_rgb: np.ndarray, surround_imgs: list = None):
        if not self.model:
            return None
        from PIL import Image
        history_str = "\n".join([f"- {h}" for h in history[-5:]])
        prompt = f"""
        You are controlling a differential-drive robot. Choose the NEXT action (one step) to achieve: "{instruction}".
        History (recent): 
        {history_str}
        Allowed actions: scan, turn_left_30, turn_right_30, move_forward_short, approach, stop.
        If target is visible, prefer approach with target name. If lost, use scan or turn towards remembered direction.
        Output JSON only: {{"action":"<name>","target":"<target_optional>","reason":"<brief>"}}
        """
        contents = [prompt]
        if img_rgb is not None:
            contents.append("Current View:")
            contents.append(Image.fromarray(img_rgb))
        if surround_imgs:
            contents.append("Surround View (Front, Right, Back, Left):")
            for img in surround_imgs:
                contents.append(Image.fromarray(img))
        try:
            self.last_prompt = prompt
            response = self.model.generate_content(
                contents,
                generation_config={"response_mime_type": "application/json"}
            )
            self.last_response = response.text
            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️ ER fallback plan error: {e}")
            return None


class ERProgressMonitor:
    """Use ER to estimate progress/stall status."""
    def __init__(self, model_name: str = VISION_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.last_prompt = None
        self.last_response = None
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            try:
                self.model = genai.GenerativeModel(self.model_name)
                print(f"✅ ER progress monitor: {self.model_name}")
            except Exception as e:
                print(f"⚠️ ER progress init error: {e}")

    def estimate(self, instruction: str, history: list, frames: list):
        if not self.model:
            return None
        from PIL import Image
        history_str = "\n".join([f"- {h}" for h in history[-5:]])
        prompt = f"""
        Estimate task progress toward: "{instruction}"
        Recent actions:
        {history_str}
        Return JSON only: {{"status": "IN_PROGRESS|STALLED|COMPLETED", "progress": 0..1, "comment": "<brief>"}}
        """
        contents = [prompt]
        for frame in frames:
            contents.append(Image.fromarray(frame))
        try:
            self.last_prompt = prompt
            response = self.model.generate_content(
                contents,
                generation_config={"response_mime_type": "application/json"},
            )
            self.last_response = response.text
            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️ ER progress error: {e}")
            return None


class VisionSystem:
    """Wrapper for Robotics-ER Model"""

    def __init__(self, model_name: str = VISION_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.last_prompt = None
        self.last_response = None

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
        Detect ONLY the {target_description} (red vertical cylinder target) in the image.
        Ignore other objects such as blue blocks or non-red items.
        Return a JSON array. Each element must have key "box_2d" with [ymin, xmin, ymax, xmax] (all 0-1000 normalized).
        If no target is found, return [].
        Example: [{{"box_2d":[200,300,800,400]}}]
        """
        self.last_prompt = prompt

        try:
            pil_img = Image.fromarray(img_rgb)
            response = self.model.generate_content(
                [prompt, pil_img],
                generation_config={"response_mime_type": "application/json"},
            )
            self.last_response = response.text
            data = json.loads(response.text)
            box = None
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and "box_2d" in first:
                    box = first["box_2d"]
            elif isinstance(data, dict) and "box_2d" in data:
                # backward compatibility
                box = data["box_2d"]

            if box:
                print(f"[VISION] Found {target_description}: {box}")
                return box
            print(f"[VISION] {target_description} not found.")
            if self.last_prompt:
                self._log(f"[PROMPT][VISION] {self.last_prompt}")
            if self.last_response:
                self._log(f"[RESPONSE][VISION] {self.last_response}")
            return None
        except Exception as e:
            print(f"⚠️ Vision detect error: {e}")
            if self.last_prompt:
                self._log(f"[PROMPT][VISION] {self.last_prompt}")
            if self.last_response:
                self._log(f"[RESPONSE][VISION] {self.last_response}")
            return None


class RobotController:
    """Control logic from ai_robot_er.py"""

    def __init__(self, turn_calibration=None):
        self.center_x = 500
        self.kp = 0.05
        self.base_speed = 4.0 # Forward speed
        self.turn_speed = 1.25
        # Discrete turn parameters (can be overridden by calibration file)
        if turn_calibration:
            self.discrete_turn_speed = float(turn_calibration.get("speed", DEFAULT_TURN_SPEED))
            self.discrete_turn_duration = float(turn_calibration.get("duration", DEFAULT_TURN_DURATION))
        else:
            self.discrete_turn_speed = DEFAULT_TURN_SPEED
            self.discrete_turn_duration = DEFAULT_TURN_DURATION

    def set_speeds(self, forward_speed=None, turn_speed=None):
        if forward_speed is not None:
            self.base_speed = float(forward_speed)
        if turn_speed is not None:
            self.turn_speed = float(turn_speed)
        print(f"⚙️ Speeds set to: Forward={self.base_speed}, Turn={self.turn_speed}")

    def decide_action(self, bbox, action_type="approach"):
        # Discrete Actions (Open-loop with duration)
        # Returns: (left_motor, right_motor), duration_seconds
        turn_speed = max(self.discrete_turn_speed, SCAN_TURN_SPEED)  # Use calibrated turn speed
        
        if action_type == "turn_left_30":
            # Stronger turn to ensure perceptible yaw change
            return [-turn_speed, turn_speed], self.discrete_turn_duration
        elif action_type == "turn_right_30":
            return [turn_speed, -turn_speed], self.discrete_turn_duration
        elif action_type == "move_forward_short":
            return [self.base_speed, self.base_speed], 1.0
        elif action_type == "stop":
            return [0.0, 0.0], 0.0

        # Closed-loop (Visual Servoing)
        if action_type == "approach":
            if not bbox:
                return [0.0, 0.0], 0.0 # Stop if lost

            ymin, xmin, ymax, xmax = bbox
            obj_center_x = (xmin + xmax) / 2.0
            obj_height = ymax - ymin
            near_center = abs(obj_center_x - self.center_x) < 120
            near_left_edge = xmin < 50
            near_right_edge = xmax > 950
            not_edge = not (near_left_edge or near_right_edge)

            # Reached condition
            if obj_height > 950 and near_center and not_edge:
                print("🎯 Target Reached!")
                return [0.0, 0.0], 0.0 # Done

            # Slow down when very close to avoid jitter
            if obj_height > 850:
                speed = max(1.5, self.base_speed * 0.4)
            elif obj_height > 700:
                speed = max(2.5, self.base_speed * 0.65)
            else:
                speed = self.base_speed

            # If the target is very close to a screen edge, reduce turning gain to avoid wild spins
            kp = 0.05
            if obj_height > 700 and (near_left_edge or near_right_edge):
                kp = 0.02
                speed = min(speed, 2.0)

            # P-Control
            error_x = obj_center_x - self.center_x
            turn = error_x * kp
            # If almost centered, prefer straight movement and longer step
            if abs(error_x) < 80:
                turn = 0.0
            # Clamp turn when very close to avoid spin
            turn = np.clip(turn, -6.0, 6.0)

            left = speed + turn
            right = speed - turn
            
            # print(f"🔍 [CTRL] cx={obj_center_x:.1f}, err={error_x:.1f}, turn={turn:.1f}, L/R={left:.1f}/{right:.1f}")
            duration = 0.2 if abs(error_x) < 80 else 0.1
            return [np.clip(left, -20, 20), np.clip(right, -20, 20)], duration # Longer step if centered
        
        return [0.0, 0.0], 0.0


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
        self.setWindowTitle("Robotics-ER Agentic Dashboard")

        # Components
        self.env = SimulationEnv()
        self.vision = VisionSystem()
        self.planner = AgentPlanner()
        self.fallback_planner = ERPlannerAssist()
        self.progress_monitor = ERProgressMonitor()
        calib = load_turn_calibration()
        self.controller = RobotController(turn_calibration=calib)
        if calib:
            print(f"✅ Loaded turn calibration: speed={calib['speed']}, duration={calib['duration']}")
        # Logging
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_file = logs_dir / f"run_{timestamp}.log"
        self.last_state_log = time.time()
        
        # State
        self.ctrl_left = 0.0
        self.ctrl_right = 0.0
        self.running = False
        self.vision_busy = False
        self.vision_thread = None
        self.vision_job = None
        self.bbox_display = None
        self.last_api_time = 0.0
        self.api_interval = 1.5
        self.current_target = None
        self.last_progress_check = 0.0
        self.progress_interval = 6.0 # seconds
        self.edge_bbox_count = 0
        self.last_motion_check = time.time()
        self.last_pose_for_motion = None
        self.skip_motion_until = 0.0
        
        # Task Management
        self.history = []
        self.current_action = None
        self.action_start_time = 0.0
        self.action_duration = 0.0
        self.thinking = False
        self.waiting_for_vision = False  # True while an async vision call is in flight
        
        # Scanning State
        self.scanning = False
        self.scan_step = 0 # 0: Front, 1: Right, 2: Back, 3: Left
        self.surround_images = [] # List of 4 images
        
        self._setup_ui()
        self._setup_timer()
        self._refresh_once()

    def _log_debug(self, msg):
        if DEBUG_MODE:
            self._log(f"[DBG] {msg}")

    def _log_action_debug(self, action_name, motors, duration, extra=""):
        if not DEBUG_MODE:
            return
        state = self.env.get_robot_state(self.ctrl_left, self.ctrl_right)
        bbox_txt = f"bbox={self.bbox_display}" if self.bbox_display else "bbox=None"
        self._log(
            f"[DBG] action={action_name} motors=L{motors[0]:+.1f}/R{motors[1]:+.1f} "
            f"dur={duration:.2f}s pos=({state.position[0]:+.2f},{state.position[1]:+.2f}) "
            f"yaw={state.yaw_deg:+.1f} {bbox_txt} {extra}".strip()
        )

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

        self.btn_start = QPushButton("Start Agent")
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        
        self.btn_randomize = QPushButton("Randomize Poles")
        self.combo_scene = QComboBox()
        self.combo_scene.addItem("scene.xml")
        self.combo_scene.addItem("scene_obstacle.xml")
        self.combo_scene.addItem("scene_complex.xml")
        self.combo_scene.addItem("scene_multi_poles.xml")
        self.combo_scene.addItem("scene_maze.xml")
        self.combo_scene.addItem("scene_clutter.xml")
        self.combo_scene.addItem("scene_calibration.xml")
        
        self.edit_instruction = QLineEdit("Go to the red pole")
        self.edit_instruction.setPlaceholderText("Enter instruction")
        
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_randomize.clicked.connect(self.on_randomize)
        self.combo_scene.currentTextChanged.connect(self.on_scene_changed)

        control_layout.addWidget(QLabel("Scene:"))
        control_layout.addWidget(self.combo_scene)
        control_layout.addWidget(QLabel("Instruction:"))
        control_layout.addWidget(self.edit_instruction)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_randomize)

        # 4. Log List
        self.list_log = QListWidget()
        self.list_log.setStyleSheet("background-color: #303030; color: white;")
        layout.addWidget(self.list_log, 0, 2, 2, 1) # Right side column

        layout.addWidget(control_panel, 2, 0, 1, 3)
        
        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 0)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 1)
        
        self.resize(1200, 700)

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
        
        # Agent Loop: Look -> Think -> Act
        if self.running:
            now = time.time()
            
            # 1. Check if current action is done
            if self.current_action and self.current_action != "approach":
                elapsed = now - self.action_start_time
                if elapsed >= self.action_duration:
                    prev_action = self.current_action
                    self.ctrl_left = 0.0
                    self.ctrl_right = 0.0
                    self.current_action = None
                    self._log_action_debug(f"{prev_action}_done", [0.0, 0.0], 0.0, "duration elapsed")
                    self.thinking = False # Ready to think again
            
            # 2. If idle and not thinking, decide next step
            if (not self.scanning) and (not self.current_action) and (not self.thinking):
                self.thinking = True
                self.statusBar().showMessage("Thinking...")
                
                # Async Planning
                # For simplicity in this demo, we might block briefly or use a thread.
                # Let's use a simple blocking call for now, but with processEvents to keep UI alive.
                # Ideally this should be in a QThread.
                QApplication.processEvents()
                
                instruction = self.edit_instruction.text()
                step = self.planner.decide_next_step(instruction, self.history, img_robot, self.surround_images)
                if step is None and self.fallback_planner:
                    self.statusBar().showMessage("Planner fallback (ER)...")
                    QApplication.processEvents()
                    step = self.fallback_planner.plan(instruction, self.history, img_robot, self.surround_images)
                    if step is None and self.fallback_planner and self.fallback_planner.last_response:
                        self._log(f"[PROMPT][ER-Fallback] {self.fallback_planner.last_prompt}")
                        self._log(f"[RESPONSE][ER-Fallback] {self.fallback_planner.last_response}")
                
                if step:
                    action_name = step.get("action")
                    target = step.get("target")
                    reason = step.get("reason")
                    
                    self._log(f"🤖 {action_name} ({reason})")
                    self.history.append(f"{action_name}: {reason}")
                    
                    if action_name == "stop":
                        self.running = False
                        self.statusBar().showMessage("Goal Reached!")
                        self._log("✅ Goal Reached!")
                    elif action_name == "scan":
                        self.scanning = True
                        self.scan_step = 0
                        self.surround_images = []
                        self.current_action = "scan"
                        self.action_start_time = now
                        self.action_duration = 0.0 # Immediate transition to scan logic
                        self.statusBar().showMessage("Scanning...")
                        self._log("🔄 Scanning surroundings...")
                    elif action_name == "approach":
                        # Need vision for approach (keep servoing until target reached/lost)
                        self.current_action = "approach"
                        self.current_target = target
                        self.waiting_for_vision = True
                        self.action_start_time = now
                        self.action_duration = 0.0
                        self._log_action_debug("approach_init", [self.ctrl_left, self.ctrl_right], 0.0, f"target={target}")
                        
                        # Reset motion baseline and skip stall detection briefly
                        self.last_pose_for_motion = (time.time(), self.env.get_robot_state(self.ctrl_left, self.ctrl_right))
                        self.skip_motion_until = time.time() + 3.0
                        if not self.vision_busy:
                             self._start_vision_job(img_robot, target)
                    else:
                        # Discrete Action
                        motors, duration = self.controller.decide_action(None, action_name)
                        self.ctrl_left, self.ctrl_right = float(motors[0]), float(motors[1])
                        self.current_action = action_name
                        self.action_start_time = now
                        self.action_duration = duration
                        self._log_action_debug(action_name, motors, duration)
                        self.statusBar().showMessage(f"Executing: {action_name}")
                else:
                    self._log("❌ Planning Failed")
                    self.running = False

        # Handle Scanning Logic
        if self.scanning:
            # We need to capture 4 images: Front(0), Right(90), Back(180), Left(270)
            # Logic: Capture -> Turn 90 -> Wait -> Capture...
            # Clear bbox display when starting scan
            self.bbox_display = None
            self.edge_bbox_count = 0
            
            # Check if we are currently turning
            if self.current_action == "scanning_turn":
                if time.time() - self.action_start_time >= self.action_duration:
                    # Turn finished, stop and capture next
                    self.ctrl_left = 0.0
                    self.ctrl_right = 0.0
                    self.current_action = None
                    # Ready to capture next step
            
            if self.current_action is None:
                # Capture current view
                self.surround_images.append(img_robot.copy())
                self.scan_step += 1
                
                if self.scan_step >= 4:
                    # Scan complete
                    self.scanning = False
                    self.thinking = False # Ready to plan with new images
                    self._log("✅ Scan Complete")
                    # Reset to front view? Or just stay? Let's stay for now.
                    # Actually, we should probably return to original orientation or just let planner handle it.
                    # Planner will know we are at "Left" view (270 deg from start).
                else:
                    # Start turn for next view (90 deg right)
                    self.current_action = "scanning_turn"
                    self.ctrl_left = SCAN_TURN_SPEED
                    self.ctrl_right = -SCAN_TURN_SPEED
                    self.action_duration = SCAN_TURN_DURATION
                    self.action_start_time = time.time()
                    self._log_action_debug("scanning_turn", [self.ctrl_left, self.ctrl_right], self.action_duration, f"step={self.scan_step}")

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

        # Continuous vision polling during approach
        if self.running and self.current_action == "approach":
            now = time.time()
            if (not self.vision_busy) and (now - self.last_api_time >= self.api_interval):
                self.waiting_for_vision = True
                self.last_api_time = now
                self._start_vision_job(img_robot, self.current_target)

        # Periodic telemetry logging for post-run analysis
        if time.time() - self.last_state_log >= 1.0:
            self.last_state_log = time.time()
            self._log_state_snapshot()

        # Motion stall detection (no pose change) -> trigger scan
        if self.running and not self.scanning and not self.waiting_for_vision and not self.vision_busy:
            now = time.time()
            if self.last_pose_for_motion is None:
                self.last_pose_for_motion = (now, self.env.get_robot_state(self.ctrl_left, self.ctrl_right))
            elif now - self.last_motion_check >= 3.0:
                prev_t, prev_state = self.last_pose_for_motion
                cur_state = self.env.get_robot_state(self.ctrl_left, self.ctrl_right)
                dist = np.linalg.norm(cur_state.position[:2] - prev_state.position[:2])
                dyaw = abs(cur_state.yaw_deg - prev_state.yaw_deg)
                self.last_motion_check = now
                self.last_pose_for_motion = (now, cur_state)
                if dist < 0.05 and dyaw < 5.0 and now >= self.skip_motion_until:
                    self._log("[STALL] Minimal motion detected -> scanning to recover.")
                    self.ctrl_left = 0.0
                    self.ctrl_right = 0.0
                    self.current_action = None
                    self.scanning = True
                    self.scan_step = 0
                    self.surround_images = []
                    self.action_start_time = time.time()
                    self.action_duration = 0.0
                    self.statusBar().showMessage("Motion stall -> scanning")

        # Progress/stall monitoring (lightweight, ER-based)
        if self.running and not self.scanning and (time.time() - self.last_progress_check >= self.progress_interval):
            self.last_progress_check = time.time()
            if self.progress_monitor and self.progress_monitor.model:
                result = self.progress_monitor.estimate(self.edit_instruction.text(), self.history, [img_robot])
                if result:
                    status = result.get("status")
                    progress = result.get("progress")
                    comment = result.get("comment", "")
                    self._log(f"🧭 Progress: {status} ({progress}) {comment}")
                    # If stalled, force stop current action and trigger scan
                    if status == "STALLED":
                        self.ctrl_left = 0.0
                        self.ctrl_right = 0.0
                        self.current_action = None
                        self.scanning = True
                        self.scan_step = 0
                        self.surround_images = []
                        self.action_start_time = time.time()
                        self.action_duration = 0.0
                        self.statusBar().showMessage("Progress stalled -> scanning")

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
        self.waiting_for_vision = False
        # Filter edge-hugging huge boxes (likely false positive)
        if bbox:
            ymin, xmin, ymax, xmax = bbox
            height = ymax - ymin
            near_edge = xmin < 50 or xmax > 950
            if near_edge and height > 900:
                self.edge_bbox_count += 1
                if self.edge_bbox_count >= 2:
                    bbox = None
                    self.bbox_display = None
                    self._log("[VISION] Dropped edge-hugging bbox as likely false positive.")
            else:
                self.edge_bbox_count = 0
        else:
            self.edge_bbox_count = 0
        
        if self.current_action == "approach":
            # We are in the middle of an approach step
            motors, duration = self.controller.decide_action(bbox, "approach")
            self.ctrl_left, self.ctrl_right = float(motors[0]), float(motors[1])
            self._log_action_debug("approach", motors, duration, "vision_update")
            
            # If target reached or lost, we finish this step
            if motors == [0.0, 0.0]:
                self.current_action = None
                self.current_target = None
                self.thinking = False
                self.ctrl_left = 0.0
                self.ctrl_right = 0.0
                self.statusBar().showMessage("Approach complete or target lost")

    def _log(self, msg):
        self.list_log.addItem(msg)
        self.list_log.scrollToBottom()
        print(msg)
        # Persist log to file for post-run analysis
        try:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        except Exception:
            pass

    def _log_state_snapshot(self):
        state = self.env.get_robot_state(self.ctrl_left, self.ctrl_right)
        bbox_txt = self.bbox_display if self.bbox_display else "None"
        scene_name = getattr(self.env, "xml_path", "unknown")
        msg = (
            f"[STATE] action={self.current_action or 'Idle'} "
            f"pos=({state.position[0]:+.2f},{state.position[1]:+.2f}) "
            f"yaw={state.yaw_deg:+.1f} "
            f"motors=L{self.ctrl_left:+.2f}/R{self.ctrl_right:+.2f} "
            f"bbox={bbox_txt} "
            f"scene={scene_name}"
        )
        self._log(msg)

    def on_start(self):
        self.running = True
        self.history = []
        self.list_log.clear()
        self.current_target = None
        self._log("▶️ Agent Started")

    def on_stop(self):
        self.running = False
        self.ctrl_left = 0.0
        self.ctrl_right = 0.0
        self.current_action = None
        self.current_target = None
        self.thinking = False
        self._log("⏹️ Agent Stopped")

    def on_randomize(self):
        self.env.randomize_poles()
        self._log("🎲 Poles Randomized")

    def on_scene_changed(self, text):
        self.env.reload_scene(text)
        self._refresh_once()
        self._log(f"Scene changed to {text}")

    def _on_vision_finished(self):
        self.vision_busy = False

    def closeEvent(self, event):
        print("Closing application, cleaning up threads...")
        try:
            if self.vision_thread and self.vision_thread.isRunning():
                self.vision_thread.quit()
                self.vision_thread.wait()
        except RuntimeError:
            pass
        event.accept()

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
        
        task_str = self.current_action if self.current_action else "Thinking..."
        if not self.running:
            task_str = "Idle"
            
        text = (
            f"<b>Status:</b> {'RUNNING' if self.running else 'STOPPED'}<br>"
            f"<b>Action:</b> {task_str}<br>"
            f"<b>Robot Pos:</b> ({state.position[0]:.2f}, {state.position[1]:.2f})<br>"
            f"<b>Yaw:</b> {state.yaw_deg:.1f} deg<br>"
            f"<b>Motors:</b> L={self.ctrl_left:.1f}, R={self.ctrl_right:.1f}<br>"
            f"<b>Vision:</b> {VISION_MODEL_NAME}<br>"
            f"<b>Planner:</b> {PLANNER_MODEL_NAME}<br>"
            f"<b>Speed:</b> FWD={self.controller.base_speed}, TURN={self.controller.turn_speed}"
        )
        self.label_info.setText(text)
    def on_instruction_changed(self, text):
        pass # Handled by Plan button

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
