# DeepNeuralPrior TP for UCBL DISS ImageMachineLearning Class

Project demonstrating Deep Image Prior variants (Gaussian denoising, inpainting, noise reconstruction) implemented with a tiny U-Net.

## Files
- [deep_neural_prior_gaussian.py](deep_neural_prior_gaussian.py) — Gaussian denoising.
- [deep_neural_prior_inpainting.py](deep_neural_prior_inpainting.py) — Inpainting with mask.
- [deep_neural_prior_noise_reconstruction.py](deep_neural_prior_noise_reconstruction.py) — Noise to image reconstruction.
- [tp_ml.yml](tp_ml.yml) — Conda environment specification for reproducible dependencies.

## Overview
Each script defines a UNet class and optimizes network parameters to reconstruct a target image its respective input.

## Requirements
Use the provided conda environment file to install exact dependencies:

```sh
conda env create -f tp_ml.yml
conda activate tp_ml
```

## Helpful Links
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [Conv2d and ConvTransposed2d](https://indico.cern.ch/event/996880/contributions/4188468/attachments/2193001/3706891/ChiakiYanagisawa_20210219_Conv2d_and_ConvTransposed2d.pdf)
- [UNet Tutorial](https://www.kaggle.com/code/akshitsharma1/unet-architecture-explained-in-one-shot-tutorial/notebook)
