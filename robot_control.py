import mujoco
import numpy as np
import cv2
import time

# --- 意思決定（Geminiの脳） ---
def gemini_planner(step_count):
    """
    ロボットの行動計画
    """
    # 動きのサイクルを定義 (1サイクル = 500ステップ)
    cycle = step_count % 500
    
    # 1. 前進 (0〜250ステップ)
    if cycle < 250:
        # 速度を 2.0 -> 15.0 にアップ！
        # ※マイナスが前進方向
        return [-15.0, -15.0]
    
    # 2. 右旋回 (250〜350ステップ)
    elif cycle < 350:
        # 左右逆回転でその場旋回
        return [8.0, -8.0]
        
    # 3. 一時停止 (350〜500ステップ)
    else:
        return [0.0, 0.0]

# --- メイン処理 ---
# 1. セットアップ
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# レンダリング設定
robot_renderer = mujoco.Renderer(model, height=240, width=320)
global_renderer = mujoco.Renderer(model, height=480, width=640)

print("=== シミュレーション開始 (ESCキーで終了) ===")

step = 0
while True:
    # 2. 物理シミュレーション実行
    # AIの思考サイクルよりも物理演算のサイクルの方が速いため、
    # 数ステップに1回だけAI(gemini_planner)を呼ぶのが一般的ですが、
    # 今回は簡易的に毎回呼び出します。
    
    motor_ctrl = gemini_planner(step)
    
    # 指令値をセット
    data.ctrl[0] = motor_ctrl[1] # 右モーター
    data.ctrl[1] = motor_ctrl[0] # 左モーター
    
    mujoco.mj_step(model, data)
    
    # 3. 画面表示 (描画は重いので5ステップに1回だけ行う)
    if step % 5 == 0:
        # ロボット視点
        robot_renderer.update_scene(data, camera="robot_cam")
        img_robot = robot_renderer.render()
        
        # 全体視点
        global_renderer.update_scene(data, camera="global_cam")
        img_global = global_renderer.render()
        
        # BGR変換(OpenCV用)
        show_robot = cv2.cvtColor(img_robot, cv2.COLOR_RGB2BGR)
        show_global = cv2.cvtColor(img_global, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Robot Eye', show_robot)
        cv2.imshow('Global View', show_global)
        
        # キー入力受付
        if cv2.waitKey(1) == 27: # ESCキー
            break

    step += 1

cv2.destroyAllWindows()