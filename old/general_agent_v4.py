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

# =========================================================
# ★ モデル設定: 用途に合わせて切り替えてください
# =========================================================
# A. ロボティクス特化 (空間認識が得意、プレビュー権限が必要)
MODEL_NAME = 'models/gemini-robotics-er-1.5-preview'

# B. 最新汎用モデル (高速、指示理解が得意。ERが使えない場合はこちら)
# MODEL_NAME = 'gemini-1.5-flash' 

print(f"🚀 使用モデル: {MODEL_NAME}")

# ---------------------------------------------------------
# Vision System: 座標取得
# ---------------------------------------------------------
class VisionSystem:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel(MODEL_NAME)
        except:
            print(f"⚠️ モデル {MODEL_NAME} が見つかりません。Standard Modelに切り替えます。")
            self.model = genai.GenerativeModel('gemini-1.5-flash')

    def detect_object(self, img_array, target_description):
        """
        物体検出を行い、正規化座標(0-1000)を返す
        """
        prompt = f"""
        Detect the "{target_description}".
        Return JSON with key "box_2d" [ymin, xmin, ymax, xmax] (0-1000 normalized).
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
            # print(f"Vision Error: {e}")
            return None

# ---------------------------------------------------------
# ★ Controller: 挙動制御ロジック (修正版)
# ---------------------------------------------------------
class RobotController:
    def __init__(self):
        self.center_x = 500        # 画面中央
        self.align_threshold = 80  # 中央とみなす範囲(±80)
        
        # ★ 速度パラメータ (プラスを前進に変更)
        self.base_speed = 15.0    # 直進時の基本速度
        self.turn_speed = 8.0     # 旋回時の速度
        self.kp = 0.02            # 旋回微調整のゲイン

    def decide_action(self, bbox, task_action):
        """
        視覚情報から「次のモーター指令」を決定する (ルールベースAI)
        """
        # 1. 見つからない場合 -> 旋回して探す
        if not bbox:
            print("  State: SEARCHING (回転探索)")
            return [self.turn_speed, -self.turn_speed], False

        ymin, xmin, ymax, xmax = bbox
        obj_center_x = (xmin + xmax) / 2.0
        obj_height = ymax - ymin
        
        # 画面中央との誤差 (右にあるとプラス)
        error = obj_center_x - self.center_x
        
        print(f"  Target X:{obj_center_x:.0f} (Err:{error:.0f}) H:{obj_height:.0f}", end=" ")

        # ----------------------------------------
        # タスク別ロジック
        # ----------------------------------------
        
        # A. 「向く (look_at)」タスク
        if task_action == "look_at":
            # 中央に入ったら完了
            if abs(error) < self.align_threshold:
                return [0.0, 0.0], True
            
            # 軸合わせ (Aligning)
            print("| State: ALIGNING (軸合わせ)")
            if error > 0: # 右にある -> 右旋回 (左タイヤ正転、右タイヤ逆転)
                return [self.turn_speed, -self.turn_speed], False
            else:         # 左にある -> 左旋回
                return [-self.turn_speed, self.turn_speed], False

        # B. 「移動する (move_to)」タスク
        elif task_action == "move_to":
            # 十分近づいたら完了
            if obj_height > 850:
                return [0.0, 0.0], True

            # 2段階制御: まず正面に向く、それから進む
            if abs(error) > self.align_threshold:
                print("| State: ALIGNING (向き調整)")
                # 向きが大きくずれている間はその場で旋回
                if error > 0: return [self.turn_speed, -self.turn_speed], False
                else:         return [-self.turn_speed, self.turn_speed], False
            else:
                print("| State: APPROACHING (接近)")
                # 正面なら直進 (少し左右差をつけてカーブしながら追うP制御)
                # 右にズレてる(error>0) -> 右に曲がりたい -> 左(L)を速く、右(R)を遅く
                # L = base + (error*kp), R = base - (error*kp)
                correction = error * self.kp
                left = self.base_speed + correction
                right = self.base_speed - correction
                return [left, right], False

        return [0.0, 0.0], False

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    vision = VisionSystem()
    controller = RobotController()
    
    if not os.path.exists('scene.xml'):
        print("❌ scene.xml not found.")
        return

    model_mj = mujoco.MjModel.from_xml_path('scene.xml')
    data_mj = mujoco.MjData(model_mj)
    renderer = mujoco.Renderer(model_mj, height=240, width=320)
    global_renderer = mujoco.Renderer(model_mj, height=480, width=640)

    # 簡易タスクリスト (Planner部分は省略し、動作確認に集中)
    tasks = [
        {"action": "move_to", "target": "red pole"}, # まず赤へ
        {"action": "look_at", "target": "blue box"}  # 次に青を向く
    ]
    
    current_task_idx = 0
    current_ctrl = [0.0, 0.0]
    last_api_time = 0
    bbox_display = None
    
    print("=== Robot Simulation Started ===")

    step = 0
    while True:
        # タスク管理
        if current_task_idx >= len(tasks):
            print("🎉 Mission Complete!")
            break
        task = tasks[current_task_idx]

        # 物理シミュレーション
        data_mj.ctrl[0] = current_ctrl[1] # 右
        data_mj.ctrl[1] = current_ctrl[0] # 左
        mujoco.mj_step(model_mj, data_mj)
        step += 1

        # AI制御ループ (間引き)
        if time.time() - last_api_time > 0.5: # 0.5秒間隔
            renderer.update_scene(data_mj, camera="robot_cam")
            img = renderer.render()
            
            # 1. 認識
            bbox_display = vision.detect_object(img, task["target"])
            
            # 2. 判断 & 制御
            new_ctrl, is_done = controller.decide_action(bbox_display, task["action"])
            current_ctrl = np.clip(new_ctrl, -20, 20)
            
            if is_done:
                print(f"✅ Task '{task['target']}' Done!")
                current_task_idx += 1
                current_ctrl = [0.0, 0.0]
                time.sleep(1)

            last_api_time = time.time()

        # 画面表示
        if step % 5 == 0:
            # ロボット視点
            renderer.update_scene(data_mj, camera="robot_cam")
            img_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            if bbox_display:
                h, w, _ = img_bgr.shape
                ymin, xmin, ymax, xmax = bbox_display
                # 枠描画
                cv2.rectangle(img_bgr, (int(xmin/1000*w), int(ymin/1000*h)), (int(xmax/1000*w), int(ymax/1000*h)), (0, 255, 0), 2)
                # 中心ガイド線
                cv2.line(img_bgr, (w//2, 0), (w//2, h), (0, 255, 255), 1) # 黄色い線が画面中央
                
            cv2.imshow('Robot Eye', img_bgr)
            
            # 全体視点
            global_renderer.update_scene(data_mj, camera="global_cam")
            cv2.imshow('Global View', cv2.cvtColor(global_renderer.render(), cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) == 27: break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()