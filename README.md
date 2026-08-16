# DeepSeek-Foundations

**The Mathematical Foundations of Modern Machine Learning (LLM Edition)**

An open, self-contained Colab curriculum for the mathematics that underpin modern machine learning — with a direct line of sight to the transformer models behind today's large language models. Released in the style of a DeepSeek technical report: one master notebook (the "paper") and seven workbooks (the "code release").

> **Abstract.** We introduce DeepSeek-Foundations, an open curriculum organized as one master report plus seven workbooks, each isolating one pillar of the foundation — linear algebra, calculus and automatic differentiation, probability and statistics, information theory, optimization, neural networks, and the transformer itself. Every workbook follows one rule: *each equation is followed by executable NumPy code*. The master notebook closes with a main result: a complete from-scratch NumPy implementation of a tiny GPT, trained end-to-end with manually derived backpropagation, gradient-checked against finite differences (measured: cross-entropy 3.91 → 2.28 nats, perplexity 50 → 9.78 on a 5 KB corpus).

## The central claim

An LLM is a differentiable function that maps a sequence of tokens to a probability distribution over the next token, trained by gradient descent on cross-entropy. Everything else — architectures, data pipelines, quantization, KV caches, mixture-of-experts, RLHF — is engineering that makes this one mechanism faster, larger, cheaper, or better aligned.

## The release

| # | Workbook | Runtime | Open in Colab |
|---|---|---|---|
| 00 | **The Master Report** (this curriculum's "paper" + tiny-GPT capstone) | ~45 min | [Open](https://colab.research.google.com/github/mattdani21/ml-math-foundations/blob/main/notebooks/00_DeepSeek_Foundations.ipynb) |
| 01 | Linear Algebra — vectors, matrices, SVD | 60 min | [Open](https://colab.research.google.com/github/mattdani21/ml-math-foundations/blob/main/notebooks/01_Linear_Algebra.ipynb) |
| 02 | Calculus & Automatic Differentiation — backprop from scratch | 75 min | [Open](https://colab.research.google.com/github/mattdani21/ml-math-foundations/blob/main/notebooks/02_Calculus_and_Autodiff.ipynb) |
| 03 | Probability & Statistics — distributions, Bayes, MLE | 75 min | [Open](https://colab.research.google.com/github/mattdani21/ml-math-foundations/blob/main/notebooks/03_Probability_and_Statistics.ipynb) |
| 04 | Information Theory — entropy, cross-entropy, perplexity | 60 min | [Open](https://colab.research.google.com/github/mattdani21/ml-math-foundations/blob/main/notebooks/04_Information_Theory.ipynb) |
| 05 | Optimization — gradient descent to Adam | 75 min | [Open](https://colab.research.google.com/github/mattdani21/ml-math-foundations/blob/main/notebooks/05_Optimization.ipynb) |
| 06 | Neural Networks — backprop, initialization, regularization | 90 min | [Open](https://colab.research.google.com/github/mattdani21/ml-math-foundations/blob/main/notebooks/06_Neural_Networks.ipynb) |
| 07 | The Transformer — attention, position, scaling laws | 90 min | [Open](https://colab.research.google.com/github/mattdani21/ml-math-foundations/blob/main/notebooks/07_Transformers.ipynb) |

**Suggested order:** 00 → 01 → 02 → 03 → 04 → 05 → 06 → 07, one pillar per sitting (30–40 min/day). Each workbook links forward to the next.

## Prerequisites

None beyond high-school math and a working `import numpy`. Every workbook opens with its own refresher; every exercise has a solution.

## The papers ladder

This curriculum exists so that frontier papers become readable cold. The natural next rungs:

Vaswani et al. 2017 (*Attention Is All You Need*) → RoPE → FlashAttention → LoRA → DeepSeek-V2 (MLA) → DeepSeek-V3 (MoE + FP8) → DeepSeek-R1 (RL reasoning).

Every one of them is standard transformer math plus well-documented engineering.

## Reproducibility

Every notebook runs entirely on a free Google Colab CPU runtime — NumPy only, no downloads, no API keys. The master report's tiny GPT trains in under a minute on CPU.

## License

MIT — see [LICENSE](LICENSE). The training corpus in the master notebook is an original essay written for this project.

---

*If you find this curriculum useful in your own study or writing, please cite it as:*

> **DeepSeek-Foundations: The Mathematical Foundations of Modern Machine Learning (LLM Edition).** Open curriculum, v1.0, August 2026. https://github.com/mattdani21/ml-math-foundations
