# 離散ターンのキャリブレーション手順

1. `python calibration_runner.py` を実行し、`calibration_turn.json` を生成する（探索レンジは speed 6–10, duration 1.5–3.0 に拡大済み）。
2. 生成された JSON には推奨 `turn_speed` と `turn_duration` が含まれる。Yaw 変化が足りない場合は speed/duration を手動で上げて再生成。
3. `er_dashboard.py` は起動時に `calibration_turn.json` を自動読込し、離散ターンに適用する（ファイルが無い場合は既定値を使用）。
4. 必要に応じて複数の速度/時間を試し、Yaw変化が目標（例: 約30度）に最も近い組み合わせを選ぶ。
5. 調整後は `pytest tests/test_robot_controller.py` でキャリブ値の適用を確認する。
