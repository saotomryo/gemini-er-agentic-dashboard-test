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

# ---------------------------------------------------------
# 1. Planner: 自然言語をタスクリストに分解する
# ---------------------------------------------------------
class TaskPlanner:
    def __init__(self):
        # テキスト処理には安価で高速なFlashを使用
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def plan_tasks(self, instruction):
        """
        入力: "赤いポールに行ってから、青い箱を向いて"
        出力: [{"action": "move_to", "target": "red pole"}, {"action": "look_at", "target": "blue box"}]
        """
        prompt = f"""
        You are a robot instruction parser. Convert the following natural language command into a sequence of tasks.
        
        Command: "{instruction}"
        
        Available actions:
        - "move_to": Approach the target until close.
        - "look_at": Turn towards the target but do not approach closely.
        
        Output JSON format:
        [
          {{"action": "move_to", "target": "description of object"}},
          ...
        ]
        Return ONLY the JSON array.
        """
        response = self.model.generate_content(prompt)
        try:
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except:
            print("⚠️ 計画の生成に失敗しました")
            return []

# ---------------------------------------------------------
# 2. Vision: 指定された物体を探す (汎用)
# ---------------------------------------------------------
class VisionSystem:
    def __init__(self):
        # 空間認識には Robotics-ER (なければPro/Flash)
        model_name = 'models/gemini-robotics-er-1.5-preview'
        # model_name = 'gemini-1.5-flash' # エラー時はこちら
        self.model = genai.GenerativeModel(model_name)

    def detect_object(self, img_array, target_description):
        """
        画像と「探すべきものの名前」を受け取り、座標を返す
        """
        prompt = f"""
        Detect the "{target_description}" in the image.
        Return the 2D bounding box in JSON format with keys "box_2d" [ymin, xmin, ymax, xmax].
        If not found, return null.
        """
        try:
            pil_img = Image.fromarray(img_array)
            response = self.model.generate_content(
                [prompt, pil_img],
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(response.text)
            if "box_2d" in data and data["box_2d"]:
                return data["box_2d"]
            return None
        except Exception as e:
            return None

# ---------------------------------------------------------
# 3. Controller: 座標をモーター指令に変換する
# ---------------------------------------------------------
class RobotController:
    def __init__(self):
        self.screen_center = 500 # 0-1000の中央
        self.kp_turn = 0.05      # 旋回ゲイン
    
    def calculate_command(self, bbox, action_type):
        """
        座標とアクションタイプ("move_to" or "look_at")から速度を計算
        戻り値: (左右モーター速度, 完了したかどうかのフラグ)
        """
        if not bbox:
            return [5.0, -5.0], False # 見つからないなら旋回探索

        ymin, xmin, ymax, xmax = bbox
        center_x = (xmin + xmax) / 2.0
        height = ymax - ymin # 物体の大きさ(近さ)

        # 画面中央とのズレ
        error_x = center_x - self.screen_center
        turn = error_x * self.kp_turn

        # --- アクション別の制御 ---
        
        # A. 近づく (Move To)
        if action_type == "move_to":
            if height > 850: # 十分近づいたら完了
                return [0.0, 0.0], True
            
            # 前進 + 旋回
            base_speed = -15.0
            return [base_speed + turn, base_speed - turn], False

        # B. 向くだけ (Look At)
        elif action_type == "look_at":
            # 中央付近に来たら完了
            if abs(error_x) < 50: 
                return [0.0, 0.0], True
            
            # その場旋回のみ
            return [turn, -turn], False
        
        return [0.0, 0.0], False

# ---------------------------------------------------------
# メイン実行部
# ---------------------------------------------------------
def main():
    # 各モジュールのインスタンス化
    planner = TaskPlanner()
    vision = VisionSystem()
    controller = RobotController()
    
    # --- シミュレータ設定 ---
    model_mj = mujoco.MjModel.from_xml_path('scene.xml')
    data_mj = mujoco.MjData(model_mj)
    renderer = mujoco.Renderer(model_mj, height=240, width=320)
    global_renderer = mujoco.Renderer(model_mj, height=480, width=640)

    # ==========================================
    # ★ ユーザーからの指示 (ここを変えると動きが変わる)
    # ==========================================
    user_instruction = "赤いポールまで移動して。そのあと、青い箱の方を向いて。"
    
    print(f"🗣️ 指示: {user_instruction}")
    print("🧠 計画中...")
    task_queue = planner.plan_tasks(user_instruction)
    print(f"📋 タスクリスト: {json.dumps(task_queue, indent=2, ensure_ascii=False)}")
    
    # 実行ループ用変数
    current_task_idx = 0
    last_api_time = 0
    current_ctrl = [0.0, 0.0]
    bbox_display = None
    
    step = 0
    while True:
        # --- タスク管理 ---
        if current_task_idx < len(task_queue):
            task = task_queue[current_task_idx]
        else:
            print("🎉 全タスク完了！")
            break # ループ終了

        # --- 物理シミュレーション ---
        data_mj.ctrl[0] = current_ctrl[1]
        data_mj.ctrl[1] = current_ctrl[0]
        mujoco.mj_step(model_mj, data_mj)
        step += 1

        # --- AI処理 (間隔を空けて実行) ---
        if time.time() - last_api_time > 1.5:
            
            renderer.update_scene(data_mj, camera="robot_cam")
            img = renderer.render()
            
            # 1. 今のターゲットを探す
            target_name = task["target"]
            print(f"👁️ 探索中: {target_name} ({task['action']})...", end="\r")
            bbox_display = vision.detect_object(img, target_name)
            
            # 2. モーター指令を計算 & 完了判定
            new_ctrl, is_done = controller.calculate_command(bbox_display, task["action"])
            current_ctrl = np.clip(new_ctrl, -20, 20) # クリップ
            
            if is_done:
                print(f"\n✅ タスク完了: {target_name}")
                current_task_idx += 1
                current_ctrl = [0.0, 0.0] # 一旦停止
                bbox_display = None
                time.sleep(1) # わかりやすく少し待つ

            last_api_time = time.time()

        # --- 画面描画 ---
        if step % 5 == 0:
            renderer.update_scene(data_mj, camera="robot_cam")
            img_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            
            # 認識枠の表示
            if bbox_display:
                h, w, _ = img_bgr.shape
                ymin, xmin, ymax, xmax = bbox_display
                cv2.rectangle(img_bgr, (int(xmin/1000*w), int(ymin/1000*h)), (int(xmax/1000*w), int(ymax/1000*h)), (0, 255, 0), 2)
                cv2.putText(img_bgr, task["target"], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            cv2.imshow('Robot Eye', img_bgr)
            
            global_renderer.update_scene(data_mj, camera="global_cam")
            cv2.imshow('Global View', cv2.cvtColor(global_renderer.render(), cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) == 27: break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()