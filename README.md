# gan-metrics
Lots of evaluation metrics of Generative Adversarial Networks in pytorch

**Work In Progress...**

## Requirements

* Python 3.x
* torch 1.x
* torchvision 0.4.x
* numpy
* tqdm

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
| Frechet Inception Distance (FID) | | |
| SSIM & PSNR | | |
| Perceptual Path Length (PPL) | | |
| Learned Perceptual Image Patch Similarity (LPIPS) | [arXiv](https://arxiv.org/abs/1801.03924) | |
| Amazon's Mechanical Turk (AMT) | [Amazon](https://www.mturk.com/) | |

## Author

[Hyeongchan Kim](http://kozistr.tech) / @kozistr
