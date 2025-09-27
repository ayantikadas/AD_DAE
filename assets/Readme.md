# AD-DAE: Diffusion Autoencoder for Unsupervised Modeling of Longitudinal Alzheimer’s Disease Progression

## Architecture
<p align="center">
  <img src="assets/architecture_ad_dae.png" alt="AD-DAE Architecture" width="800">
</p>

> Figure: High-level schematic of the AD-DAE framework showing the encoder-guided diffusion decoder, latent shift estimation, and consistency guidance.

## Inference
- A walkthrough notebook is provided: **`Inference_AD_DAE.ipynb`**  
- It demonstrates:
  - Loading a trained AD-DAE checkpoint
  - Encoding a baseline scan
  - Estimating latent shifts from progression attributes (age, cognitive status)
  - Generating follow-up images via the diffusion decoder
  - Optional evaluation/visualization steps

### Quick Start
1. Clone the repo and install dependencies (see `requirements.txt`).
2. Place your trained weights under `checkpoints/` (or update the path in the notebook).
3. Open and run **`Inference_AD_DAE.ipynb`** cell-by-cell.

## Repo Structure (excerpt)
