# Paper Summary: Bootstrap Your Own Latent (BYOL)

**Grill et al., NeurIPS 2020**
**arXiv:** [2006.07733](https://arxiv.org/abs/2006.07733)

---

## Problem
Self-supervised visual representation learning methods — most notably SimCLR and MoCo — achieve strong performance by contrasting positive pairs against large sets of negative pairs. This contrastive design introduces two entangled requirements: (1) large batch sizes or memory banks to supply enough negatives, and (2) careful augmentation design to avoid trivially distinguishable negatives. BYOL asks whether negative pairs are necessary at all.

## Method
BYOL trains two networks jointly. The **online network** (parameterized by θ) consists of an encoder $f_\theta$, a projector $g_\theta$, and a predictor $q_\theta$. The **target network** (parameterized by ξ) mirrors the encoder and projector but has no predictor. Target parameters ξ are an exponential moving average (EMA) of online parameters θ — they are never trained by gradient descent.

Given an image $x$, two stochastic augmentations produce views $v$ and $v'$. The online network processes $v$ through all three components to produce prediction $q_\theta(z_\theta)$, where $z_\theta = g_\theta(f_\theta(v))$. The target network processes $v'$ through its encoder and projector to produce target projection $z'_\xi = g_\xi(f_\xi(v'))$. Both vectors are L2-normalized, and the loss is:

$$\mathcal{L}_{\theta,\xi} = \left\| \bar{q}_\theta(z_\theta) - \bar{z}'_\xi \right\|_2^2 = 2 - 2 \cdot \frac{\langle q_\theta(z_\theta), z'_\xi \rangle}{\|q_\theta(z_\theta)\| \cdot \|z'_\xi\|}$$

This is symmetrized by swapping the views and summing both losses. Only the online parameters θ receive gradients; the target is updated as $\xi \leftarrow \tau \xi + (1 - \tau)\theta$, where $\tau$ follows a cosine schedule from $\tau_{\text{base}}$ to 1.

**Three mechanisms prevent collapse:**
1. The **predictor** breaks the symmetry between online and target networks, making it harder for the model to find a degenerate fixed point.
2. The **stop-gradient** on the target prevents both networks from collapsing to a mutual trivial solution.
3. The **EMA update** provides a slowly-evolving regression target that is more stable than the online network itself, acting as a regularizer.

## Key Results
- **74.3%** top-1 accuracy on ImageNet with linear evaluation (ResNet-50), matching SimCLR v2 with 10× fewer parameters in the projection head.
- Outperforms contrastive baselines on 11 of 15 transfer learning benchmarks.
- More robust to changes in augmentation — performance degrades less when individual augmentations are removed compared to SimCLR.

## Significance
BYOL demonstrated that the self-supervised learning objective does not need to be contrastive. This was widely considered implausible before publication, since without an explicit repulsive force between embeddings, the expected outcome is representational collapse (all inputs mapped to the same vector). The paper opened an entirely new design space for self-supervised methods, directly inspiring SimSiam, VICReg, Barlow Twins, and DINO. The central question it raised — *why doesn't this collapse?* — generated substantial follow-up theoretical work and remains partially open.
