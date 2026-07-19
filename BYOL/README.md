# BYOL: Bootstrap Your Own Latent — Reproduction
**Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning**
*(Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko — DeepMind, 2020)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

---

## Project Overview
This repository contains a research-grade PyTorch reproduction of **BYOL** (Bootstrap Your Own Latent), the self-supervised learning method that achieves state-of-the-art representation quality *without* negative pairs. This is the second entry in my [ML Research Journey](https://github.com/devmulchandani), following the [SimCLR reproduction](https://github.com/devmulchandani).

The entire pipeline — pretraining, linear evaluation, k-NN evaluation, ablations, and visualizations — runs end-to-end on **Google Colab Free Tier** (T4 GPU, ≤2 hours).

## Motivation
In Week 1 (SimCLR), I identified a core limitation of contrastive learning: the method's heavy dependence on large batch sizes to supply enough negative pairs. SimCLR requires batch sizes of 4096–8192 to work well, which demands multi-TPU infrastructure that most researchers will never have access to.

BYOL answers a question I wrote in my SimCLR research notes: *"Is there a mathematical way to decouple batch size from the number of negative samples?"* The answer turns out to be more radical than expected — BYOL eliminates negative pairs entirely. Instead of contrasting positive and negative pairs, it trains an **online network** to predict the representations of a slowly-moving **target network**, using only positive pairs. The fact that this doesn't collapse to a trivial constant solution is one of the most surprising results in recent self-supervised learning.

## Paper
- **Title:** Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning
- **Authors:** Jean-Bastien Grill et al. (DeepMind)
- **Conference:** NeurIPS 2020
- **arXiv:** [2006.07733](https://arxiv.org/abs/2006.07733)
- **Official Implementation:** [google-deepmind/deepmind-research/byol](https://github.com/google-deepmind/deepmind-research/tree/master/byol) (JAX/Haiku)

## Repository Structure

```text
BYOL/
├── implementation/
│   └── BYOL_Notebook.ipynb            # Complete pretraining, eval, & ablation pipeline
├── docs/
│   ├── paper_summary.md               # ≤2 page paper summary
│   ├── research_notes.md              # Definitions, concepts, connections, questions
│   └── presentation.md                # 10-slide research presentation outline
├── results/
│   └── README.md                      # Description of expected outputs
├── reflection.md                      # Personal research diary
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Installation
```bash
pip install -r requirements.txt
```
All dependencies are pre-installed on Google Colab. The `requirements.txt` exists for local reproducibility.

## How to Run
1. Open `implementation/BYOL_Notebook.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. The notebook is fully self-contained — it downloads CIFAR-10 automatically.
4. Total runtime: ~1.5–2 hours on a T4.

## Compute Adaptations

| Dimension | Paper | This Reproduction | Reason |
|---|---|---|---|
| Encoder | ResNet-50 | ResNet-18 | Fits T4 VRAM; faster iteration |
| Dataset | ImageNet (1.28M) | CIFAR-10 (50K) | Fits Colab free tier runtime |
| Batch size | 4096 | 256 | Single-GPU memory constraint |
| Epochs | 1000 | 200 | ~2 hrs total runtime |
| Optimizer | LARS | AdamW | LARS is critical at BS=4096; AdamW is sufficient and cleaner at BS=256 |
| EMA base τ | 0.996 | 0.996 | Kept — core to the method |
| Precision | float32 | Mixed (AMP) | Memory savings + throughput on T4 |

## Results

| Metric | Value |
|---|---|
| BYOL Linear Probe (CIFAR-10) | ~88–92% |
| k-NN Evaluation (k=200) | ~85–88% |
| Supervised Baseline (ResNet-18) | ~93–95% |
| Without Predictor (collapse) | ~10% (random) |

## Research Questions
1. **Why doesn't BYOL collapse?** What is the role of the predictor, the stop-gradient, and BatchNorm?
2. **How sensitive is BYOL to the EMA decay rate?** What happens at τ=0.9 vs τ=0.999?
3. **Is the asymmetric architecture (predictor on online only) strictly necessary?**
4. **Does BYOL still work without BatchNorm?** (The "information leakage" hypothesis.)

## Future Work
- Implement **SimSiam** (Chen & He, 2021) to test the hypothesis that momentum is not even needed.
- Study **VICReg** and **Barlow Twins** as alternative non-contrastive objectives.
- Extend BYOL pretraining to **CIFAR-100** and evaluate transfer quality.
- Investigate the role of augmentation asymmetry (view1 vs view2 pipelines).

## Disclaimer
This is an educational reproduction for research study purposes. It is not affiliated with DeepMind. The implementation is written from scratch in PyTorch, guided by the original paper. The official DeepMind implementation (JAX/Haiku) was used only as a reference to verify architectural details.

## References
1. Grill, J.-B., et al. "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning." *NeurIPS*, 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
2. Chen, T., et al. "A Simple Framework for Contrastive Learning of Visual Representations." *ICML*, 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
3. He, K., et al. "Momentum Contrast for Unsupervised Visual Representation Learning." *CVPR*, 2020. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
4. Chen, X. & He, K. "Exploring Simple Siamese Representation Learning." *CVPR*, 2021. [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)
5. Richemond, P. H., et al. "BYOL Works Even Without Batch Statistics." *arXiv*, 2020. [arXiv:2010.10241](https://arxiv.org/abs/2010.10241)
