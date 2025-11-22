import mujoco
import numpy as np
import cv2
import time
import json
import os  # 追加
from dotenv import load_dotenv  # 追加
import google.generativeai as genai
from PIL import Image

# ==========================================
# 1. 環境変数の読み込み & API設定
# ==========================================
# .env ファイルを読み込む
load_dotenv()

# 環境変数からAPIキーを取得
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ エラー: APIキーが見つかりません。")
    print(".envファイルに GEMINI_API_KEY を設定してください。")
    exit(1)

genai.configure(api_key=API_KEY)

# モデル設定 (Robotics-ER プレビュー版)
# ※ アクセス権がない場合は 'gemini-1.5-pro' 等に変更してください
MODEL_NAME = 'models/gemini-robotics-er-1.5-preview'
# MODEL_NAME = 'gemini-1.5-flash' # 代替案

try:
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"✅ モデル '{MODEL_NAME}' を設定しました。")
except Exception as e:
    print(f"⚠️ モデル設定エラー: {e}")
    print("標準の 'gemini-1.5-flash' などに書き換える必要があるかもしれません。")
    # フォールバックする場合
    # model = genai.GenerativeModel('gemini-1.5-flash') 

# ==========================================
# 2. Robotics-ER 特化プロンプト
# ==========================================
PROMPT = """
Detect the red vertical pole (cylinder target) in the image.
Return a JSON array. Each element has key "box_2d" with [ymin, xmin, ymax, xmax] (all 0-1000 normalized).
If no target is found, return [].
Example output: [{"box_2d": [200, 300, 800, 400]}]
"""

# ==========================================
# 3. 制御ロジック (P制御)
# ==========================================
def calculate_motor_command(bbox_norm):
    if not bbox_norm:
        print("Searching... (Target not found)")
        return [5.0, -5.0] # 旋回探索

    ymin, xmin, ymax, xmax = bbox_norm
    
    # 重心とサイズ
    center_x = (xmin + xmax) / 2.0
    height = ymax - ymin
    
    # 画面中央(500)を目指す
    error_x = center_x - 500
    
    # 制御ゲイン
    KP = 0.05
    turn = error_x * KP
    
    # 距離判定
    if height > 900:
        print("🎯 Target Reached!")
        return [0.0, 0.0]
    
    # 前進しながら旋回補正
    base_speed = 15.0 # 前進に変更 (-15.0 -> 15.0)
    left = base_speed + turn
    right = base_speed - turn
    
    print(f"🔍 [DEBUG] cx={center_x:.1f}, err={error_x:.1f}, turn={turn:.1f}, L/R={left:.1f}/{right:.1f}")

    return [np.clip(left, -20, 20), np.clip(right, -20, 20)]

def detect_with_er_model(img_array):
    try:
        pil_img = Image.fromarray(img_array)
        
        # 生成設定 (JSONモード強制)
        response = model.generate_content(
            [PROMPT, pil_img],
            generation_config={"response_mime_type": "application/json"} 
        )
        
        data = json.loads(response.text)

        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "box_2d" in first and first["box_2d"]:
                return first["box_2d"]
        elif isinstance(data, dict) and "box_2d" in data and data["box_2d"]:
            return data["box_2d"]
        return None
            
    except Exception as e:
        print(f"⚠️ Detection Error: {e}")
        return None

# ==========================================
# 4. メインループ
# ==========================================
# XMLファイルがあるか確認
if not os.path.exists('scene.xml'):
    print("❌ エラー: 'scene.xml' が見つかりません。")
    exit(1)

model_mj = mujoco.MjModel.from_xml_path('scene.xml')
data_mj = mujoco.MjData(model_mj)

robot_renderer = mujoco.Renderer(model_mj, height=240, width=320)
global_renderer = mujoco.Renderer(model_mj, height=480, width=640)

print(f"=== {MODEL_NAME} Visual Servoing Start ===")

current_ctrl = [0.0, 0.0]
last_api_time = 0
bbox_display = None 

step = 0
while True:
    # 物理ステップ
    data_mj.ctrl[0] = current_ctrl[1]
    data_mj.ctrl[1] = current_ctrl[0]
    mujoco.mj_step(model_mj, data_mj)
    step += 1

    # API呼び出し
    if time.time() - last_api_time > 1.5: 
        
        robot_renderer.update_scene(data_mj, camera="robot_cam")
        img = robot_renderer.render()
        
        print(f"👁️ {MODEL_NAME} Scanning...", end="\r")
        bbox_display = detect_with_er_model(img)
        current_ctrl = calculate_motor_command(bbox_display)
        
        last_api_time = time.time()

    # 画面描画
    if step % 5 == 0:
        robot_renderer.update_scene(data_mj, camera="robot_cam")
        img_bgr = cv2.cvtColor(robot_renderer.render(), cv2.COLOR_RGB2BGR)
        
        if bbox_display:
            h, w, _ = img_bgr.shape
            ymin, xmin, ymax, xmax = bbox_display
            x1, y1 = int(xmin/1000*w), int(ymin/1000*h)
            x2, y2 = int(xmax/1000*w), int(ymax/1000*h)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Robot Eye', img_bgr)
        
        global_renderer.update_scene(data_mj, camera="global_cam")
        cv2.imshow('Global View', cv2.cvtColor(global_renderer.render(), cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) == 27: break

cv2.destroyAllWindows()
