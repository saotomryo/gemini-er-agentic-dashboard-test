import mujoco
import numpy as np
import cv2
import time
import json
import os
import math
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 空間推論能力が高いモデル推奨 (Robotics-ER 1.5 Preview または 2.0 Flash Exp)
MODEL_NAME = 'gemini-2.0-flash-exp' # または 'models/gemini-robotics-er-1.5-preview'

print(f"🚀 Path Planning Agent ({MODEL_NAME})")

# ---------------------------------------------------------
# 1. Path Planner: 視覚情報から「経路(Trajectory)」を生成
# ---------------------------------------------------------
class PathPlanner:
    def __init__(self):
        self.model = genai.GenerativeModel(MODEL_NAME)

    def plan_trajectory(self, img_array, goal_desc, obstacle_desc):
        """
        画像を見て、障害物を避けながらゴールに至る「経路の点(Waypoint)リスト」を返す
        """
        # 画像の下端中央(500, 1000)がロボットの現在地と仮定
        prompt = f"""
        You are a robot navigation system.
        Goal: "{goal_desc}"
        Obstacle: "{obstacle_desc}"
        
        Task:
        Plan a safe path from the bottom-center of the image (robot's current position) to the Goal.
        The path must AVOID the Obstacle.
        
        Output JSON:
        {{
          "reasoning": "Explain the spatial situation (e.g., Obstacle is in the center, so go right)",
          "waypoints": [
             [y1, x1], [y2, x2], ... [yn, xn]
          ]
        }}
        - Coordinates must be normalized integers (0-1000).
        - [y, x] format.
        - The first waypoint should be the immediate next step.
        - Generate 3 to 5 waypoints representing the curve.
        """
        try:
            pil_img = Image.fromarray(img_array)
            response = self.model.generate_content(
                [prompt, pil_img],
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(response.text)
            return data
        except Exception as e:
            # print(f"Plan Error: {e}")
            return None

# ---------------------------------------------------------
# 2. Controller: 「経路の次の点」に向かって進む (Pure Pursuit)
# ---------------------------------------------------------
class TrajectoryController:
    def __init__(self):
        self.screen_center_x = 500
        self.base_speed = 12.0
        self.turn_gain = 0.04

    def follow_path(self, waypoints):
        """
        ウェイポイントリストの「最初の点」を目指して操舵する
        """
        if not waypoints or len(waypoints) == 0:
            # 経路がない場合 -> 旋回して探す or 停止
            return [5.0, -5.0], False

        # 次の目標点 (Next Waypoint)
        # y, x = waypoints[0] (一番手前の点)
        # もし点が近すぎる(yが大きい=画面下部)なら、2つ目の点を狙うなどの工夫も可能
        target_y, target_x = waypoints[0] 
        
        # 画面中央とのズレ
        error_x = target_x - self.screen_center_x
        
        # 到達判定 (ゴールの点が画面下部=手前 に来たら完了)
        # ※厳密には最後のウェイポイントのY座標などで判定
        if len(waypoints) == 1 and target_y > 800:
            return [0.0, 0.0], True

        # 制御計算 (P制御)
        turn = error_x * self.turn_gain
        
        # 障害物を避けるために大きく曲がっている時は減速する
        current_speed = self.base_speed
        if abs(error_x) > 200: 
            current_speed *= 0.5 

        left = current_speed + turn
        right = current_speed - turn
        
        return [left, right], False

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    planner = PathPlanner()
    controller = TrajectoryController()
    
    if not os.path.exists('scene.xml'): return
    model_mj = mujoco.MjModel.from_xml_path('scene.xml')
    data_mj = mujoco.MjData(model_mj)
    renderer = mujoco.Renderer(model_mj, height=240, width=320) # AIの目
    global_renderer = mujoco.Renderer(model_mj, height=480, width=640) # 人間の目

    print("=== 障害物回避ナビゲーション開始 ===")
    print("赤いポールを目指しますが、青い箱があれば避けます。")

    current_ctrl = [0.0, 0.0]
    last_api_time = 0
    current_plan = None # 現在のAIの計画結果
    
    # サイクル設定
    API_INTERVAL = 0.8 # 経路計画は少し重いので0.8秒ごと

    step = 0
    while True:
        # 物理ステップ
        data_mj.ctrl[0] = current_ctrl[1]
        data_mj.ctrl[1] = current_ctrl[0]
        mujoco.mj_step(model_mj, data_mj)
        step += 1

        # --- AI: 認識と計画 (Re-planning) ---
        if time.time() - last_api_time > API_INTERVAL:
            renderer.update_scene(data_mj, camera="robot_cam")
            img = renderer.render()
            
            print("🧠 Planning Path...", end="\r")
            
            # 「赤いポールに行きたい、青い箱は避けて」と指示
            plan_result = planner.plan_trajectory(img, "red vertical pole", "blue box")
            
            if plan_result:
                waypoints = plan_result.get("waypoints", [])
                reason = plan_result.get("reasoning", "")
                print(f"\n🤖 考え: {reason}")
                
                # コントローラーに経路を渡して速度をもらう
                new_ctrl, is_done = controller.follow_path(waypoints)
                current_ctrl = np.clip(new_ctrl, -20, 20)
                current_plan = waypoints # 描画用に保存
                
                if is_done:
                    print("🎉 GOAL REACHED!")
                    current_ctrl = [0.0, 0.0]
                    time.sleep(2)
                    break
            else:
                # 計画失敗時はその場でゆっくり旋回
                current_ctrl = [5.0, -5.0]

            last_api_time = time.time()

        # --- 画面描画 ---
        if step % 5 == 0:
            renderer.update_scene(data_mj, camera="robot_cam")
            img_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            h, w, _ = img_bgr.shape

            # ★ AIが考えた「経路」を線で描画
            if current_plan:
                # ロボットの現在地(画面下中央)
                prev_pt = (w//2, h)
                
                for pt in current_plan:
                    # 0-1000正規化座標をピクセルに変換
                    y_norm, x_norm = pt
                    cx = int(x_norm / 1000 * w)
                    cy = int(y_norm / 1000 * h)
                    
                    # 線を引く
                    cv2.line(img_bgr, prev_pt, (cx, cy), (0, 255, 255), 2) # 黄色い線
                    cv2.circle(img_bgr, (cx, cy), 4, (0, 0, 255), -1)      # 赤い点
                    prev_pt = (cx, cy)

            cv2.imshow('Robot Eye (Path Planning)', img_bgr)
            
            global_renderer.update_scene(data_mj, camera="global_cam")
            cv2.imshow('Global View', cv2.cvtColor(global_renderer.render(), cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) == 27: break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()