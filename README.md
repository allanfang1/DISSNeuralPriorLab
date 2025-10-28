# Deep Neural Prior TP for Image Machine Learning Class of DISS program

Project demonstrating Deep Image Prior variants (Gaussian denoising, inpainting, noise reconstruction) implemented with a tiny U-Net.

## Files
- [deep_neural_prior_gaussian.py](deep_neural_prior_gaussian.py) — Gaussian denoising (uses [`MyUNet`](deep_neural_prior_gaussian.py)).
- [deep_neural_prior_inpainting.py](deep_neural_prior_inpainting.py) — Inpainting with mask (uses [`MyUNet`](deep_neural_prior_inpainting.py)).
- [deep_neural_prior_noise_reconstruction.py](deep_neural_prior_noise_reconstruction.py) — Noise to image reconstruction (uses [`MyUNet`](deep_neural_prior_noise_reconstruction.py)).
- [tp_ml.yml](tp_ml.yml) — Conda environment specification for reproducible dependencies.

## Overview
Each script defines a small U-Net class (`MyUNet`) and optimizes network parameters to reconstruct a target image from each respective input.

## Requirements
Use the provided conda environment file to install exact dependencies:

```sh
conda env create -f [tp_ml.yml](http://_vscodecontentref_/0)
conda activate tp_ml
