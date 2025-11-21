import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. モデルとデータの読み込み
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# 2. レンダラー（カメラ映像作成機）の準備
renderer = mujoco.Renderer(model, height=480, width=640)

# 3. シミュレーションループ
print("シミュレーション開始...")
frames = []

# 100ステップだけ動かしてみる
for i in range(100):
    # --- ここで Gemini API が介入する余地があります ---
    # 例: Geminiが「右に動け」と言ったら、アクションを書き換える
    # data.ctrl[0] = ... (今回は自由落下のみ)
    
    mujoco.mj_step(model, data)

    # カメラ 'eye' から見た映像を取得 (Geminiへの入力データ)
    if i % 10 == 0:  # 10ステップごとに撮影
        renderer.update_scene(data, camera="eye")
        img = renderer.render()
        frames.append(img)

# 4. 結果の確認（最後のフレームを表示）
print("シミュレーション終了。最後のフレームを表示します。")
plt.imshow(frames[-1])
plt.title("View from Robot Camera")
plt.axis('off')
plt.show()

# (オプション) 動画として保存したい場合
# import imageio
# imageio.mimsave('simulation.gif', frames, fps=30)