# Gemini Robotics-ER 1.5 概要

[公式ドキュメント](https://ai.google.dev/gemini-api/docs/robotics-overview?hl=ja) に基づく、Gemini Robotics-ER 1.5 の主要機能とアーキテクチャの要約です。

## 1. 主な機能

Gemini Robotics-ER 1.5 は、ロボットが物理世界を認識・理解し、タスクを実行するためのマルチモーダルモデルです。

*   **空間認識とコンテキスト理解**:
    *   画像、動画、テキストを入力として受け取ります。
    *   オブジェクトの位置（バウンディングボックス）や、シーン内の関係性を理解します。
*   **エージェント機能 (Orchestration)**:
    *   「リンゴをボウルに入れて」のような抽象的な指示を、具体的なサブタスクのシーケンスに分解します。
    *   論理的な手順（移動する → つかむ → 運ぶ → 置く）を生成します。
*   **構造化出力**:
    *   JSON形式で座標（バウンディングボックス、点）やラベルを出力します。
    *   `box_2d`: `[ymin, xmin, ymax, xmax]` (0-1000正規化座標)
*   **軌跡生成 (Trajectory)**:
    *   開始点から目標点までの移動経路（ウェイポイントのリスト）を生成できます。
    *   例: `[{"point": [y, x], "label": "0"}, ...]`
*   **コード実行 (Code Execution)**:
    *   Pythonコードを生成・実行して、複雑な計算やロジック（例: 画像の特定領域のズーム、アームの逆運動学計算など）を処理できます。

## 2. 実装パターン

### A. 基本的な物体検出 (Visual Servoing)
現在の `er_dashboard.py` で実装しているパターンです。
1.  **入力**: 現在のカメラ画像 + 「赤いポールを見つけて」
2.  **出力**: バウンディングボックス
3.  **制御**: ボックスの中心に向かって移動（PID制御など）

### B. エージェント型 (Plan & Execute)
ユーザーが要望している「計画 → 実行 → 修正」のパターンです。

1.  **Planning (計画)**:
    *   ユーザー指示: 「赤いポールのところに行ってから、青い障害物を避けて」
    *   モデル出力 (JSON): `[{"action": "goto", "target": "red pole"}, {"action": "avoid", "target": "blue obstacle"}]`
2.  **Execution (実行)**:
    *   リストの先頭からタスクを取り出し、実行します。
3.  **Monitoring & Correction (監視と修正)**:
    *   実行中、常にカメラ画像で状況を確認します。
    *   **修正**: ターゲットが見つからない、または状況が変わった場合（例: 障害物が動いた）、再度モデルに画像を見せて「計画の修正」を依頼します。

## 3. 推奨されるアーキテクチャ

```mermaid
graph TD
    User[ユーザー指示] --> Planner[Planner (Gemini)]
    Planner -->|Action Plan| Manager[Task Manager]
    Manager -->|Current Task| Executor[Executor]
    
    subgraph Control Loop
        Executor -->|Image + Task| Vision[Vision (Gemini)]
        Vision -->|BBox / Status| Controller[Robot Controller]
        Controller -->|Motor Cmd| Robot[Robot (MuJoCo)]
        Robot -->|New Image| Executor
    end
    
    Executor -.->|Failure / Change| Planner
```
