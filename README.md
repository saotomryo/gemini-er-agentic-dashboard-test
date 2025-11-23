# Robotics-ER Dashboard

Gemini Robotics-ER を使った自律ロボット制御デモ。トップレベルの実行スクリプトは `er_dashboard.py`（Qt UI）です。それ以外のドキュメントやモジュールはサブフォルダに整理しています。

## 構成
- `er_dashboard.py` / `ai_robot_er.py`: 実行スクリプト
- `src/`: 共有モジュール（`tool_defs.py`, `geom_utils.py` など）
- `docs/`: 仕様・サマリ・状態遷移図など（`specifications.md`, `robotics_er.txt`, `state_machine.md` ほか）
- `tests/`: 単体テスト
- `logs/`: 実行ログ（`run_*.log`）
- シーンXML: `scenes/scene*.xml`

## 使い方（概要）
- 依存をインストール後、`python er_dashboard.py` でQt版ダッシュボードを起動。
- OpenCV版デモは `python sim_dashboard.py` 。
- ログは `logs/run_*.log` に保存されます。デバッグは `ER_DEBUG=1` を環境変数で指定。

詳しくは `docs/` の各ドキュメントを参照してください。 README は随時更新します。  
