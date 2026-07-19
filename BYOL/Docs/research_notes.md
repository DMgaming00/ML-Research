# Research Notes: BYOL

## Problem Definition
Contrastive self-supervised methods (SimCLR, MoCo) learn representations by pulling positive pairs together and pushing negative pairs apart. This requires either massive batch sizes (SimCLR: up to 8192) or auxiliary memory banks (MoCo: queue of 65536 keys). Both solutions add significant engineering complexity and compute cost. BYOL asks: *is the repulsive term even necessary, or is the attractive term alone sufficient to learn non-trivial representations?*

## Previous Approaches
1. **SimCLR (Chen et al., 2020):** Simplified contrastive learning to augmentation + encoder + MLP + NT-Xent loss. Requires batch sizes 4096+ to supply enough in-batch negatives. Performance degrades rapidly at small batch sizes.
2. **MoCo v1/v2 (He et al., 2020):** Introduced a momentum-updated encoder and a queue of past representations to decouple the number of negatives from batch size. More memory-efficient but adds architectural complexity (queue management, shuffled BN for multi-GPU).
3. **SwAV (Caron et al., 2020):** Combined contrastive learning with online clustering (Sinkhorn-Knopp). Multi-crop strategy reduces compute while improving performance. Still relies on a form of negative contrast through the clustering assignment.
4. **DeepCluster (Caron et al., 2018):** Alternated between k-means clustering and training. Early evidence that non-contrastive objectives could work, but suffered from cluster degeneracy and required periodic re-initialization.

## Main Contributions of BYOL
1. **Negative-free self-supervised learning:** Achieves SOTA without any explicit negative pair mechanism — no contrastive loss, no memory bank, no clustering.
2. **Asymmetric architecture:** The online network has a predictor MLP that the target network lacks. This architectural asymmetry, combined with stop-gradient on the target, is essential for preventing collapse.
3. **EMA target network:** The target parameters are an exponential moving average of the online parameters, providing a slowly-evolving regression target that stabilizes training.
4. **Augmentation robustness:** BYOL is less sensitive to the choice of augmentations than contrastive methods, suggesting it learns more robust features rather than features that are merely contrastive.

## Key Concepts

### Online vs. Target Network
- **Online network:** encoder $f_\theta$ → projector $g_\theta$ → predictor $q_\theta$. This is the only network that receives gradients.
- **Target network:** encoder $f_\xi$ → projector $g_\xi$. No predictor. Parameters ξ are updated *only* via EMA: $\xi \leftarrow \tau\xi + (1-\tau)\theta$.
- The predictor exists only on the online side. This breaks the symmetry and is critical — without it, the system collapses.

### Exponential Moving Average (EMA)
- $\xi \leftarrow \tau\xi + (1-\tau)\theta$
- τ starts at $\tau_{\text{base}} = 0.996$ and increases toward 1.0 via a cosine schedule over training.
- Early in training (τ ≈ 0.996), the target updates relatively quickly to track the online network's rapid initial learning.
- Late in training (τ → 1.0), the target becomes nearly frozen, providing a very stable regression target.
- The cosine schedule: $\tau_k = 1 - (1 - \tau_{\text{base}}) \cdot (\cos(\pi k / K) + 1)/2$

### The Collapse Problem
Why would anyone expect a positive-only loss to work? Without negatives, the trivial solution is: map every input to the same constant vector. Both networks output $\mathbf{c}$, the loss $\| \bar{\mathbf{c}} - \bar{\mathbf{c}} \|^2 = 0$, and the model has "learned" nothing. Three mechanisms prevent this:
1. **Stop-gradient:** The target doesn't receive gradients, so it can't collude with the online network to find a mutual trivial fixed point.
2. **Predictor asymmetry:** The online network must predict the target's output through an additional MLP, making it harder to settle on a degenerate solution.
3. **EMA momentum:** The target moves slowly, acting as a moving regression target that the online network must continuously chase.

### The BatchNorm Controversy
Richemond et al. (2020) argued that BatchNorm implicitly introduces a form of contrastive signal by making each sample's representation depend on other samples in the batch. This "information leakage" could serve as an implicit negative mechanism. However, they later showed that BYOL works even without batch statistics (using group norm or layer norm), suggesting that BatchNorm helps but is not the sole explanation.

## My Observations
- **BYOL is bootstrapping in the truest sense.** The target network provides "pseudo-labels" (its own representations) that the online network learns to predict. As the online network improves, the EMA updates improve the target, which provides better pseudo-labels. This is a self-improving loop — very similar in spirit to EM algorithms or self-training in semi-supervised learning.
- **The predictor acts as an implicit regularizer.** Because the predictor adds parameters that the target doesn't have, the online network can't trivially copy the target. The predictor must learn a meaningful transformation from online to target space.
- **Augmentation asymmetry matters.** View 1 gets Gaussian blur with p=1.0, no solarization. View 2 gets blur with p=0.1, solarization with p=0.2. This forces the two views to be meaningfully different, not just stochastically different.

## Research Questions While Reading
- *"If the predictor prevents collapse, why can't we put a predictor on both sides?"* — This would restore symmetry. Chen & He (SimSiam) later showed that symmetrized with stop-gradient actually works, suggesting the predictor's role is subtler than simple asymmetry.
- *"What happens if we initialize the target randomly and never EMA-update it?"* — The target provides a random but fixed regression target. This actually works poorly but doesn't fully collapse, suggesting the EMA is important for quality but not strictly for collapse prevention.
- *"How does BYOL relate to EM?"* — Grill et al. draw this connection explicitly. The online update is like the M-step (maximize given current targets), and the EMA update is like a soft E-step (update the assignment/target).
- *"Is the L2 normalization strictly necessary, or would raw MSE work?"* — Without normalization, the model can trivially minimize the loss by shrinking all representations toward zero. L2 normalization projects everything onto the unit sphere, eliminating magnitude as a degree of freedom.

## Connections to Other Papers
| Paper | Relationship to BYOL |
|---|---|
| SimCLR (Chen et al., 2020) | BYOL directly removes SimCLR's negative pair requirement |
| MoCo (He et al., 2020) | BYOL borrows the momentum encoder idea but drops the contrastive loss and queue |
| SimSiam (Chen & He, 2021) | Shows that BYOL works even without momentum — stop-gradient alone suffices |
| Barlow Twins (Zbontar et al., 2021) | Alternative non-contrastive approach: decorrelation of embedding dimensions |
| VICReg (Bardes et al., 2022) | Explicitly decomposes the non-collapse condition into variance, invariance, and covariance terms |
| DINO (Caron et al., 2021) | Extends the teacher-student EMA framework to Vision Transformers with self-distillation |

## Future Directions
- Understanding collapse prevention through the lens of spectral analysis of the representation covariance matrix.
- Extending BYOL to multi-modal settings (image-text, image-audio).
- Studying whether BYOL-style objectives produce more calibrated uncertainty estimates than contrastive methods.
- Scaling laws for non-contrastive self-supervised learning: how does representation quality scale with compute?
