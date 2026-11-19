# CycleGAN for Unpaired Image-to-Image Translation

> Course: Deep Learning · Semester 2

## Overview

Implementation of a cycle-consistent adversarial network for translating between two image domains without paired examples. The generator reuses a frozen VGG16 feature extractor; the training loop uses adversarial loss plus cycle-consistency and a decaying learning-rate schedule.

## Repository structure

- `notebooks/cyclegan-vgg16.ipynb` - model definition and custom training loop.
- `docs/cyclegan-report.pdf` - report on the apple-to-orange translation experiment.

## Method

- `tfds.load('cycle_gan/horse2zebra')`, normalization to [-1, 1], random cropping and flipping.
- VGG16 up to `block5_pool` as a frozen extractor.
- U-Net-style generator with skip connections, `LayerNormalization`, GELU and conditioning convolutions.
- Four-class discriminator (real/fake for each domain) with sparse categorical cross-entropy.
- Adversarial loss plus reconstruction (cycle) loss; AdamW with decaying LR; `@tf.function` and `GradientTape`.

## Important note about the data

The notebook uses the **horse2zebra** dataset from TensorFlow Datasets, while the accompanying report describes a separate **apple2orange** run. They are related implementations with different domains; the README keeps them together but explicitly distinguishes them.

## Requirements

`tensorflow`, `tensorflow-datasets`, `numpy`, `matplotlib` and `tqdm`.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/cyclegan-vgg16.ipynb
```

The dataset is downloaded automatically by `tfds` on the first run.

---
Author: Karla Maria Monroy Alvarez
