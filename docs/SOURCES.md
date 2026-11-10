# Sources and adaptations

## Implementation provenance

Reference: Nathan Inkawhich, **DCGAN Tutorial**, PyTorch Tutorials.

- Tutorial: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- Inspected source revision: `c703e69c9f323ca87e3f6bcca537d926252d9806`
- Source: https://github.com/pytorch/tutorials/blob/c703e69c9f323ca87e3f6bcca537d926252d9806/beginner_source/dcgan_faces_tutorial.py
- License: BSD 3-Clause, retained in `THIRD_PARTY_LICENSES.txt`.

The reference uses 64×64 CelebA images, convolutional/transposed-convolutional layers, batch normalization, Gaussian latent input, normal weight initialization, alternating updates, and Adam at 0.0002 with beta1 0.5. Our implementation adapts the spatial progression to 32×32, reduces hidden generator channels to 256/128/64, supports grayscale/RGB, and replaces explicit sigmoid plus BCE with equivalent BCEWithLogitsLoss. We add point-model MLPs, configurations, provenance capture, resume support, evaluation, and Colab execution. This adapts established methodology and makes no novelty claim.

## Papers

1. Goodfellow et al. (2014), *Generative Adversarial Nets*. https://arxiv.org/abs/1406.2661
2. Radford, Metz, and Chintala (2015, revised 2016), *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. https://arxiv.org/abs/1511.06434

## Datasets and runtime

- MNIST: LeCun, Cortes, and Burges. https://yann.lecun.org/exdb/mnist/ ; loader: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
- Fashion-MNIST: Xiao, Rasul, and Vollgraf (2017). https://arxiv.org/abs/1708.07747 ; official files: https://github.com/zalandoresearch/fashion-mnist
- KMNIST: Clanuwat et al. (2018), *Deep Learning for Classical Japanese Literature*. https://arxiv.org/abs/1812.01718 ; official data: https://github.com/rois-codh/kmnist
- CIFAR-10: Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*. https://www.cs.toronto.edu/~kriz/cifar.html
- Torchvision datasets: https://docs.pytorch.org/vision/stable/datasets.html
- Colab FAQ: https://research.google.com/colaboratory/faq.html . GPU availability, runtime lifetimes, and usage limits vary.

SWD here is the average empirical one-dimensional Wasserstein-1 distance over 128 fixed random unit projections. Its full implementation and sample budgets appear in the package and README. It serves as a diagnostic for these 2D distributions rather than a universal GAN score.
