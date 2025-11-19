import mujoco
import numpy as np
import cv2
import time
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = 'gemini-2.0-flash-exp' # または gemini-1.5-flash

# ---------------------------------------------------------
# Planner & Vision (ここは変更なし)
# ---------------------------------------------------------
class TaskPlanner:
    def __init__(self):
        self.model = genai.GenerativeModel(MODEL_NAME)
    def plan_tasks(self, instruction):
        prompt = f"""
        Instruction: "{instruction}"
        Output strictly JSON array of tasks with keys "action" (move_to/look_at) and "target".
        Example: [{{"action": "move_to", "target": "red pole"}}]
        """
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text)
        except: return []

class VisionSystem:
    def __init__(self):
        self.model = genai.GenerativeModel(MODEL_NAME)
    def detect_object(self, img_array, target_description):
        prompt = f"""
        Find "{target_description}". Return JSON {{ "box_2d": [ymin, xmin, ymax, xmax] }} (0-1000).
        If not found, return null.
        """
        try:
            pil_img = Image.fromarray(img_array)
            response = self.model.generate_content([prompt, pil_img], generation_config={"response_mime_type": "application/json"})
            data = json.loads(response.text)
            if "box_2d" in data and data["box_2d"]: return data["box_2d"]
            return None
        except: return None

# ---------------------------------------------------------
# ★ RobotController: ご指摘の動作ロジックを実装
# ---------------------------------------------------------
class RobotController:
    def __init__(self):
        self.screen_center_x = 500 # 画面中央(X)
        self.center_threshold = 100 # 中央とみなす許容範囲(±100)
        
        # ゲイン調整
        self.turn_speed = 5.0   # 旋回速度
        self.move_speed = -10.0 # 前進速度(マイナス)

    def calculate_command(self, bbox, action_type):
        """
        「中心に来るまで旋回」→「中心に来たら進む」ロジック
        """
        if not bbox:
            print("  > ターゲット捜索中(旋回)...")
            return [self.turn_speed, -self.turn_speed], False

        ymin, xmin, ymax, xmax = bbox
        center_x = (xmin + xmax) / 2.0
        height = ymax - ymin

        # 画面中央とのズレ (右にあるとプラス、左にあるとマイナス)
        error_x = center_x - self.screen_center_x
        
        # デバッグ表示
        print(f"  > Target X:{center_x:.0f} (Err:{error_x:.0f}) Size:{height:.0f}")

        # --- 完了判定 ---
        if action_type == "move_to" and height > 850:
            return [0.0, 0.0], True
        elif action_type == "look_at" and abs(error_x) < self.center_threshold:
            return [0.0, 0.0], True

        # --- 動作決定 (State Machine) ---
        
        # Case 1: ターゲットが正面にない場合 -> 「旋回モード」
        if abs(error_x) > self.center_threshold:
            # ターゲットが右(error > 0) -> 右回転したい
            # 右回転: 左タイヤ前進(マイナス)、右タイヤ後退(プラス)
            if error_x > 0:
                print("  > 右旋回で軸合わせ")
                return [-self.turn_speed, self.turn_speed], False
            else:
                print("  > 左旋回で軸合わせ")
                return [self.turn_speed, -self.turn_speed], False

        # Case 2: ターゲットが正面にある場合 -> 「前進モード」
        else:
            if action_type == "move_to":
                print("  > 正面捕捉！前進")
                # まっすぐ進む
                return [self.move_speed, self.move_speed], False
            else:
                # look_atタスクなら、正面に向いた時点でほぼ完了だが微調整
                return [0.0, 0.0], True

        return [0.0, 0.0], False

# ---------------------------------------------------------
# メイン実行部
# ---------------------------------------------------------
def main():
    planner = TaskPlanner()
    vision = VisionSystem()
    controller = RobotController()
    
    if not os.path.exists('scene.xml'): return
    model_mj = mujoco.MjModel.from_xml_path('scene.xml')
    data_mj = mujoco.MjData(model_mj)
    renderer = mujoco.Renderer(model_mj, height=240, width=320)
    global_renderer = mujoco.Renderer(model_mj, height=480, width=640)

    # 指示
    user_instruction = "赤いポールまで行って。そのあと、青い箱の方を向いて。"
    print(f"🗣️ 指示: {user_instruction}")
    
    task_queue = planner.plan_tasks(user_instruction)
    print(f"📋 Plan: {json.dumps(task_queue, ensure_ascii=False)}\n")
    
    current_task_idx = 0
    last_api_time = 0
    current_ctrl = [0.0, 0.0]
    bbox_display = None
    API_INTERVAL = 0.5 

    step = 0
    while True:
        if current_task_idx >= len(task_queue):
            print("🎉 全タスク完了！")
            break

        task = task_queue[current_task_idx]

        # 物理ステップ
        data_mj.ctrl[0] = current_ctrl[1] # 右
        data_mj.ctrl[1] = current_ctrl[0] # 左
        mujoco.mj_step(model_mj, data_mj)
        step += 1

        # AI認識 & 制御更新
        if time.time() - last_api_time > API_INTERVAL:
            renderer.update_scene(data_mj, camera="robot_cam")
            img = renderer.render()
            
            bbox_display = vision.detect_object(img, task["target"])
            new_ctrl, is_done = controller.calculate_command(bbox_display, task["action"])
            current_ctrl = np.clip(new_ctrl, -20, 20)
            
            if is_done:
                print(f"✅ {task['target']} 完了！")
                current_task_idx += 1
                current_ctrl = [0.0, 0.0]
                bbox_display = None
                time.sleep(1)

            last_api_time = time.time()

        # 描画
        if step % 5 == 0:
            renderer.update_scene(data_mj, camera="robot_cam")
            img_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            if bbox_display:
                h, w, _ = img_bgr.shape
                ymin, xmin, ymax, xmax = bbox_display
                cv2.rectangle(img_bgr, (int(xmin/1000*w), int(ymin/1000*h)), (int(xmax/1000*w), int(ymax/1000*h)), (0, 255, 0), 2)
                # 中心線を描画してわかりやすくする
                cv2.line(img_bgr, (w//2, 0), (w//2, h), (100, 100, 100), 1)

            cv2.imshow('Robot Eye', img_bgr)
            global_renderer.update_scene(data_mj, camera="global_cam")
            cv2.imshow('Global View', cv2.cvtColor(global_renderer.render(), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == 27: break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()