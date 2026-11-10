# GAN theory and implementation

The generator maps latent noise z to G(z), inducing a distribution p_g. The discriminator D estimates whether an input came from the real distribution rather than the generator. The original minimax game is

\[
\min_G\max_D V(D,G)=E_{x\sim p_{data}}\log D(x)+E_{z\sim p_z}\log(1-D(G(z))).
\]

For fixed G, maximizing the integrand pointwise gives D*(x) = p_data(x)/(p_data(x)+p_g(x)), where the denominator is positive. Substitution gives V(D*,G) = −log(4) + 2 JS(p_data || p_g). The ideal global optimum has p_g = p_data and D* = 1/2 on their support. Infinite-capacity/function-space optimization assumptions do not guarantee convergence of finite networks with alternating stochastic updates. D near 1/2 can also indicate an undertrained discriminator.

## Practical losses

The discriminator produces logits s(x), with D(x) = sigmoid(s(x)). Its loss sums real-label-one and fake-label-zero binary cross entropies. Generated samples are detached during this update. The generator minimizes the non-saturating loss L_G = −E_z log D(G(z)), implemented as label-one cross entropy on generated samples. This gives stronger gradients when D confidently rejects fake samples than the saturating minimax G objective.

Discriminator weights are frozen during G's backward pass, while gradients flow through D's computation to G. The image discriminator retains training-mode batch normalization, matching the tutorial's alternating training behavior. `BCEWithLogitsLoss` combines sigmoid and cross entropy for numerical stability.

## Interpreting results

Synthetic geometry allows direct comparisons. Mode coverage alone can miss probability between modes, so report occupancy and in-mode fraction alongside SWD. Projected distances cannot fully describe every distributional difference.

Image grids show recognizable structure, artifacts, and obvious repetition. They cannot establish full class coverage or absence of memorization. We use unconditional models and single-seed image runs, limiting reliability claims.

Losses describe a changing game and do not rank datasets by quality. Under the ideal matched-distribution discriminator, the summed D loss is about 2 log(2), while the non-saturating G loss is about log(2).

Sources: Goodfellow et al. (2014), https://arxiv.org/abs/1406.2661 ; Radford et al. (2015/2016), https://arxiv.org/abs/1511.06434 ; PyTorch DCGAN tutorial, https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html .
