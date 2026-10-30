# 🤖 miku.ai

<div align="center">

**Building a GPT-like Large Language Model from scratch**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%2012.4)-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-In%20Development-F7DF1E?style=for-the-badge&logoColor=black)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

> *"A GPT-2 architecture copy... but significantly more humble (and way more stupid)."*  
> — A phantom project.

</div>

---

## Overview

**miku.ai** is a hands-on exploration of building, training, and understanding transformer-based Large Language Models (LLMs) completely from scratch.

Following the principles from [*Build a Large Language Model (From Scratch)* by Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch), this repository dives beneath high-level abstractions like Hugging Face wrappers to implement every fundamental component manually:

- Custom text pre-processing and tokenization algorithms
- Dataset pipelines and sliding-window context generators
- Embeddings, multi-head self-attention mechanisms, and transformer blocks
- Autoregressive next-token prediction and text generation

---

## Architecture & Pipeline

```text
Raw Text Corpus (the-verdict.txt)
          │
          ▼
┌───────────────────────────┐
│   Tokenizer (v1)          │  ➔ Regex split + Vocab mapping + <unk> fallback
└───────────────────────────┘
          │
          ▼
┌───────────────────────────┐
│   Sliding Window Context  │  ➔ Input (x) vs Target (y) pairs:
└───────────────────────────┘      x: ["I", "HAD", "always", "thought"]
          │                        y: ["HAD", "always", "thought", "Jack"]
          ▼
┌───────────────────────────┐
│   PyTorch Dataset/Loader  │  ➔ Batching & tensor formatting (WIP)
└───────────────────────────┘
          │
          ▼
┌───────────────────────────┐
│   Transformer & Attention │  ➔ Multi-Head Self-Attention + GPT-2 Blocks (WIP)
└───────────────────────────┘
```

---

## Project Structure

```bash
miku.ai/
├── data/                      # Training corpora and benchmark texts
│   └── the-veredict.txt       # Sample training story (Edith Wharton)
├── src/
│   ├── tokenizator.py         # Custom regex-based tokenizer (Tokennizator_v1)
│   ├── dataset.py             # PyTorch Dataset & DataLoader implementation
│   └── main.py                # Pipeline execution & tokenization demo
├── requirements.txt           # Project dependencies (PyTorch, tiktoken, etc.)
└── readme.md                  # Project documentation
```

---

## Getting Started

### 1. Prerequisites

- **Python 3.10+**
- (Optional) NVIDIA GPU with **CUDA 12.4+** support for accelerated PyTorch training.

### 2. Installation

Clone the repository and prepare a virtual environment:

```bash
# Clone the repository
git clone https://github.com/<your-username>/miku.ai.git
cd miku.ai

# Create & activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Demo

Run the tokenization and sliding-window context extraction pipeline:

```bash
python -m src.main
```

**Sample Output:**

```text
Total tokens: 4690
Context window (x): [53, 44, 149, 1003]  --> "I HAD always thought"
Target window  (y): [44, 149, 1003, 57]  --> "HAD always thought Jack"
```

---

## Roadmap & Milestones

- [x] **Phase 1: Tokenization Fundamentals**
  - [x] Regex-based text cleaner & vocabulary generator
  - [x] Encode / decode functions with `<unk>` token handling
  - [x] Sliding context window pairs ($x \rightarrow y$)
- [ ] **Phase 2: Data Pipeline & Subword Encoding**
  - [ ] Byte-Pair Encoding (BPE) integration with `tiktoken`
  - [ ] PyTorch `Dataset` and `DataLoader` with batching & padding
- [ ] **Phase 3: Transformer Architecture**
  - [ ] Token & Positional embeddings
  - [ ] Multi-Head Self-Attention with causal masking
  - [ ] FeedForward network, LayerNorm, and GELU activations
  - [ ] Transformer block residual connections
- [ ] **Phase 4: Training & Sampling**
  - [ ] Cross-entropy loss calculation & training loop
  - [ ] Temperature-scaled autoregressive text generation

---

## References & Acknowledgments

- **Sebastian Raschka** — [*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch)
- **OpenAI** — [*Language Models are Unsupervised Multitask Learners* (GPT-2 Paper)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

<div align="center">
  <sub>Built with ❤️ and curiosity.</sub>
</div>
