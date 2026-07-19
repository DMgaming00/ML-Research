# Research Reflections: BYOL Reproduction

**Name:** Dev Mulchandani
**Week:** 2
**Paper:** BYOL (Grill et al., NeurIPS 2020)

## What Surprised Me

The biggest surprise was that removing negative pairs didn't just *work* — it worked *better*. I came into this reproduction with strong priors from SimCLR. In Week 1, I internalized the idea that contrastive learning fundamentally needs negatives to create a repulsive force that prevents collapse. The NT-Xent denominator — that sum over all negatives — felt like the structural core of why these methods work. Removing it felt like removing load-bearing walls from a building.

But BYOL demonstrates that the repulsive force was doing double duty: preventing collapse *and* introducing a dependence on batch size that constrained the entire method. By finding an alternative collapse-prevention mechanism (predictor asymmetry + stop-gradient + EMA), BYOL keeps the representation quality while removing the batch size constraint. In retrospect, this makes sense — the useful signal was always in the positive pair. The negatives were a regularizer, not a teacher.

## What Confused Me

The collapse question genuinely confused me for days. I kept trying to formalize *why* the predictor prevents collapse and kept going in circles. If the predictor is just an MLP, and the loss is MSE between two L2-normalized vectors, then the trivial solution where both networks output the same constant vector *still makes the loss zero*. So why doesn't it collapse?

The key insight I eventually reached is that it's not any single mechanism — it's the interaction between all three: stop-gradient, predictor, and EMA momentum. Remove any one of them, and you either get collapse or severely degraded performance. The stop-gradient prevents the target from colluding with the online network. The predictor adds capacity that the target doesn't have, making it harder to settle on a symmetric degenerate solution. And the EMA ensures the target is always slightly different from the online network, creating a moving target that the online network must continuously adapt to.

I also found it interesting that even after Chen & He's SimSiam paper showed that momentum isn't strictly necessary (stop-gradient + predictor alone suffices), the *quality* of the learned representations is still better with momentum. The EMA target seems to act as implicit ensembling or regularization.

## Biggest Lesson

The technical lesson is that **not all components of a loss function are essential to the learning signal**. SimCLR's negatives felt essential; they weren't. This generalizes to a broader research principle: question every component of a method, especially the ones that seem load-bearing. The components that seem most fundamental are often the ones worth removing first.

The engineering lesson is that implementing the EMA update correctly — making sure it's `@torch.no_grad()`, making sure you're iterating over paired parameters, making sure the cosine schedule is right — is deceptively tricky. The math is simple ($\xi \leftarrow \tau\xi + (1-\tau)\theta$), but getting it right in PyTorch requires careful attention to parameter ordering and gradient flow.

## Biggest Misconception

I assumed that the predictor's role was to "project" the online representation into the target's space — that the online and target projectors would drift apart over training, and the predictor would learn a mapping between them. This is wrong. Because the target is an EMA of the online network, they share approximately the same parameters. The predictor's role isn't spatial alignment — it's breaking the architectural symmetry so that collapse is not a stable equilibrium.

## Connection to Week 1 (SimCLR)

Reproducing BYOL after SimCLR felt like watching a researcher solve a problem I identified. In my SimCLR research notes, I wrote: *"They rely on batch sizes of up to 8192. How on earth do I run this on a single academic GPU?"* and noted that this question leads to BYOL. Living through that intellectual progression — from understanding the problem to implementing the solution — is exactly the kind of learning trajectory that reading papers alone can't replicate.

The other connection is the augmentation pipeline. BYOL reuses almost exactly the same augmentations as SimCLR (random crop, color jitter, grayscale, Gaussian blur) but adds two things: solarization on view 2, and an asymmetry between the two views' augmentation probabilities. This was a detail I would have missed without implementing both methods.

## Next Steps
- Implement **SimSiam** to test the claim that EMA is not necessary.
- Study **VICReg** and **Barlow Twins** as alternative non-contrastive approaches.
- Investigate whether BYOL-style pretraining transfers effectively to CIFAR-100 or STL-10.
