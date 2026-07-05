<div align="center">

# PDDAI

**ゼロから構築したTransformer言語モデル — 469万パラメータ**

[🇬🇧 English](README-en.md) • [🇸🇦 العربية](README-ar.md) • [🇨🇳 中文](README-zh.md) • [🇷🇺 Русский](README-ru.md) • [🇫🇷 Français](README-fr.md) • [🇪🇸 Español](README-es.md) • [🇩🇪 Deutsch](README-de.md) • [🇵🇹 Português](README-pt.md) • [🇮🇳 हिन्दी](README-hi.md)

</div>

**469万パラメータ**のデコーダーオンリーTransformerモデル。PyTorchでゼロから構築 — GQA、RoPE、SwiGLU、Refresh Gates、KVキャッシュ推論、マルチソース検索、エージェントツール、RAG、9Router統合を搭載。

## 機能

| カテゴリ | 詳細 |
|---|---|
| **アーキテクチャ** | グループ化クエリアテンション（6クエリヘッド、3 KVヘッド）、回転位置埋め込み、SwiGLU FFN、Refresh Gates、KVキャッシュ |
| **トレーニング** | 混合精度（AMP）、勾配蓄積、Muon最適化器、Unlikelihood損失 + ラベルスムージング、チェックポイント保存 |
| **チャット** | トークン単位のストリーミング出力、終了時にトレーニングデータをエクスポート |
| **エージェント** | ツール支援推論 + マルチソース検索（Wikipedia、Wikidata、arXiv、PubMed、StackExchange、GitHub）、電卓、ファイル読み取り |
| **検索ルーター** | クエリ意図に基づく最適なAPIソースの選択と結果集約（3000文字制限） |
| **9Routerパイプライン** | 外部AI APIが検索結果を圧縮し、ローカルモデルにクリーンなコンテキストを提供 |
| **出力クリーナー** | 軽量後処理：制御文字除去、繰り返し低減、文の重複除去、スペース修正 |
| **RAG** | 検索拡張生成 — シンプル版と改善版 |
| **AI2AI** | 2モデル間の自律対話、ターンテイキングをシミュレーション |
| **トレーニングエクスポート** | 会話を9Routerでフォーマットし、高品質な訓練データとして保存 |

## クイックスタート

```bash
# ゼロからトレーニング
python main.py

# トレーニング済みモデルとチャット
python chat.py

# エージェントモード
python -m agents.agent

# AI同士の対話
python ai2ai.py
```

## プロジェクト構造

```
PDDAI/
├── config.py               # ハイパーパラメータ + 9Router設定
├── main.py                 # トレーニングパイプライン
├── chat.py                 # チャットインターフェース
├── ai2ai.py                # AI間対話
├── models/
│   └── transformer.py      # Transformerモデル
├── training/
│   └── train.py            # トレーニングループ
├── agents/
│   └── agent.py            # エージェント
├── tools/
│   └── registry.py         # ツールシステム
├── data/
│   ├── tokenizer.py        # BPEトークナイザー
│   └── dataset.py          # データセット読み込み
├── rag/
│   ├── simple_rag.py       # 基本RAG
│   └── better_rag.py       # 改善RAG
├── memory/
│   └── light_memory.py     # 軽量メモリシステム
└── utils/
    ├── auto_config.py      # ハードウェア自動検出
    └── muon.py             # Muon最適化器
```

## 要件

- Python 3.10+
- PyTorch 2.0+
- CUDA対応GPU推奨（CPUでも動作可）
- HuggingFace `datasets` ライブラリ

## アーキテクチャ

- **GQA** — 6クエリヘッド、3 KVヘッドでメモリ効率向上
- **RoPE** — 学習パラメータ不要の相対位置エンコーディング
- **SwiGLU** — ゲート付きFFN（Swish × GLU）
- **Refresh Gates** — レイヤーごとの選択的状態更新
- **KVキャッシュ** — トークンあたりO(1)の生成

## ライセンス

非商用利用のみ。[LICENSE](LICENSE) を参照。
