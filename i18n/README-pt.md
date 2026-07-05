<div align="center">

# PDDAI

**Modelo transformer construído do zero — 4,69M parâmetros**

[🇬🇧 English](README-en.md) • [🇸🇦 العربية](README-ar.md) • [🇨🇳 中文](README-zh.md) • [🇯🇵 日本語](README-ja.md) • [🇷🇺 Русский](README-ru.md) • [🇫🇷 Français](README-fr.md) • [🇪🇸 Español](README-es.md) • [🇩🇪 Deutsch](README-de.md) • [🇮🇳 हिन्दी](README-hi.md)

</div>

Um modelo transformer decoder-only de **4,69M parâmetros** construído do zero com PyTorch — com GQA, RoPE, SwiGLU, Refresh Gates, inferência com cache KV, busca multi-fontes, ferramentas de agente, RAG e integração 9Router.

## Funcionalidades

| Categoria | Detalhes |
|---|---|
| **Arquitetura** | Atenção de consulta agrupada (6 cabeças de consulta, 3 cabeças KV), embeddings posicionais rotativos, SwiGLU FFN, Refresh Gates, cache KV |
| **Treinamento** | Precisão mista (AMP), acumulação de gradientes, otimizador Muon, perda Unlikelihood + suavização de rótulos, salvamento de checkpoints |
| **Chat** | Conversa simples com saída em fluxo (token por token) e exportação de dados de treinamento ao sair |
| **Agente** | Raciocínio com ferramentas e busca multi-fontes (Wikipedia, Wikidata, arXiv, PubMed, StackExchange, GitHub), calculadora, leitor de arquivos |
| **Roteador de busca** | Roteamento inteligente de consultas para selecionar as melhores fontes e agregar resultados (limite de 3000 caracteres) |
| **Pipeline 9Router** | API de IA externa que comprime resultados de busca em contexto limpo para o modelo local |
| **Limpador de saída** | Pós-processamento leve: remoção de caracteres de controle, redução de repetições, deduplicação de frases, correção de espaçamento |
| **RAG** | Geração aumentada por recuperação — versões simples e melhorada |
| **AI2AI** | Conversa autônoma entre dois modelos com simulação de turnos |
| **Exportação de treinamento** | Conversas formatadas via 9Router e salvas como dados de treinamento de alta qualidade |

## Início rápido

```bash
# Treinar do zero
python main.py

# Conversar com o modelo treinado
python chat.py

# Agente com ferramentas
python -m agents.agent

# Conversa IA-IA
python ai2ai.py
```

## Estrutura do projeto

```
PDDAI/
├── config.py               # Hiperparâmetros + configurações 9Router
├── main.py                 # Pipeline de treinamento
├── chat.py                 # Interface de chat
├── ai2ai.py                # Conversa IA-IA
├── models/
│   └── transformer.py      # TransformerModel, GQA, RoPE, SwiGLU
├── training/
│   └── train.py            # Loop de treinamento
├── agents/
│   └── agent.py            # Agente com raciocínio + busca
├── tools/
│   └── registry.py         # Sistema de ferramentas
├── data/
│   ├── tokenizer.py        # Tokenizador BPE
│   └── dataset.py          # Carregamento de dados
├── rag/
│   ├── simple_rag.py       # RAG básico
│   └── better_rag.py       # RAG melhorado
├── memory/
│   └── light_memory.py     # Sistema de memória leve
└── utils/
    ├── auto_config.py      # Detecção automática de hardware
    └── muon.py             # Otimizador Muon
```

## Requisitos

- Python 3.10+
- PyTorch 2.0+
- GPU com CUDA recomendado (funciona em CPU)
- Biblioteca HuggingFace `datasets`

## Arquitetura

- **GQA** — 6 cabeças de consulta, 3 cabeças KV para memória eficiente
- **RoPE** — Codificação posicional relativa sem parâmetros aprendidos
- **SwiGLU** — FFN com porta (Swish × GLU)
- **Refresh Gates** — Atualizações seletivas por camada
- **Cache KV** — Geração O(1) por token

## Licença

Apenas uso não comercial. Veja [LICENSE](LICENSE).
