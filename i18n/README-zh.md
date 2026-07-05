<div align="center">

# PDDAI

**从零构建的Transformer语言模型 — 469万参数**

[🇬🇧 English](README-en.md) • [🇸🇦 العربية](README-ar.md) • [🇯🇵 日本語](README-ja.md) • [🇷🇺 Русский](README-ru.md) • [🇫🇷 Français](README-fr.md) • [🇪🇸 Español](README-es.md) • [🇩🇪 Deutsch](README-de.md) • [🇵🇹 Português](README-pt.md) • [🇮🇳 हिन्दी](README-hi.md)

</div>

一个 **469万参数** 的仅解码器Transformer模型，使用PyTorch从零构建 — 具备GQA、RoPE、SwiGLU、Refresh Gates、KV缓存推理、多源搜索、智能体工具、RAG和9Router集成。

## 特性

| 类别 | 详情 |
|---|---|
| **架构** | 分组查询注意力（6查询头，3 KV头），旋转位置编码，SwiGLU FFN，Refresh Gates，KV缓存 |
| **训练** | 混合精度（AMP），梯度累积，Muon优化器，非似然损失 + 标签平滑，检查点保存（最后/最佳/最终） |
| **聊天** | 简单对话，逐Token流式输出，退出时导出训练数据 |
| **智能体** | 工具辅助推理，多源搜索（维基百科、维基数据、arXiv、PubMed、StackExchange、GitHub），计算器，文件读取器 |
| **搜索路由** | 查询意图路由，选择最佳API源并聚合结果（3000字符限制） |
| **9Router流水线** | 外部AI API将多源搜索结果压缩为本地模型的干净上下文 |
| **输出清洗** | 轻量后处理：控制字符移除，重复减少，句子去重，间距修正 |
| **RAG** | 检索增强生成 — 简单版和改进版 |
| **AI2AI** | 双模型自主对话，模拟轮流发言 |
| **训练导出** | 对话经由9Router格式化，作为高质量训练数据保存 |

## 快速开始

```bash
# 从零训练
python main.py

# 与训练好的模型聊天
python chat.py

# 使用智能体
python -m agents.agent

# AI对AI对话
python ai2ai.py
```

## 项目结构

```
PDDAI/
├── config.py               # 超参数 + 9Router设置
├── main.py                 # 训练流水线
├── chat.py                 # 聊天界面（流式）
├── ai2ai.py                # AI对AI对话
├── models/
│   └── transformer.py      # Transformer模型
├── training/
│   └── train.py            # 训练循环
├── agents/
│   └── agent.py            # 智能体
├── tools/
│   └── registry.py         # 工具系统
├── data/
│   ├── tokenizer.py        # BPE分词器
│   └── dataset.py          # 数据集加载
├── rag/
│   ├── simple_rag.py       # 基础RAG
│   └── better_rag.py       # 改进RAG
├── memory/
│   └── light_memory.py     # 轻量记忆系统
└── utils/
    ├── auto_config.py      # 硬件自动检测
    └── muon.py             # Muon优化器
```

## 要求

- Python 3.10+
- PyTorch 2.0+
- 推荐使用支持CUDA的GPU（CPU也可运行）
- HuggingFace `datasets` 库

## 架构

- **GQA** — 6查询头，3 KV头，高效内存
- **RoPE** — 相对位置编码，无需学习参数
- **SwiGLU** — 门控FFN（Swish × GLU）
- **Refresh Gates** — 逐层选择性状态更新
- **KV缓存** — 每个Token O(1)生成

## 许可证

仅限非商业使用。参见 [LICENSE](LICENSE)。
