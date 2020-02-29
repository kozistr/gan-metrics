# gan-metrics
Lots of evaluation metrics of Generative Adversarial Networks in pytorch

**Work In Progress...**

## Requirements

* Python 3.x
* torch 1.x
* torchvision 0.4.x
* numpy
* scipy
* cython
* pot
* pyyaml

## Usage

1. `Clone` the repository and `Move` the directory.

```shell script
$ git clone https://github.com/kozistr/gan-metrics
$ cd ./gan-metrics
```

2. `Install` the `requirements.txt` 

```shell script
$ pip3 install -r requirements.txt
```

3. `Configure` the `config.yml` 

```shell script
open the file and edit
```

4. `Run`

```shell script
$ python3 -m metrics
```

## Metrics

| Metric | Paper | Code |
| :---: | :---: | :---: |
| Inception Score (IS) | [arXiv](https://arxiv.org/abs/1801.01973) | [code]( ./metrics/is) |
| Frechet Inception Distance (FID) | | [code](./metrics/fid) |
| Kernel Inception Distance (KID) | | |
| SSIM & PSNR | | |
| Precision & Recall | | |
| Earth Mover's Distance (EMD, a.k.a wasserstein distance) | [wiki](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) | [code](./metric/emd) |
| Perceptual Path Length (PPL) | | |
| Learned Perceptual Image Patch Similarity (LPIPS) | [arXiv](https://arxiv.org/abs/1801.03924) | |
| Reliable Fidelity & Diversity (PRDC) | [arXiv](https://arxiv.org/abs/2002.09797) | |
| Amazon's Mechanical Turk (AMT) | [Amazon](https://www.mturk.com/) | |

## DataSets

You can edit the `transform` function with your flavor!

* MNIST / FashionMNIST
* CIFAR10 / CIFAR100
* LSUN
* CelebA
* ImageNet
* Custom

## References

* GAN-Metrics : https://github.com/xuqiantong/GAN-Metrics
* GAN_Metrics-Tensorflow : https://github.com/taki0112/GAN_Metrics-Tensorflow

## Author

[Hyeongchan Kim](http://kozistr.tech) / @kozistr
