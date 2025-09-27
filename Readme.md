# AD-DAE: Unsupervised Modeling of Longitudinal Alzheimer’s Disease Progression with Diffusion Auto-Encoder

## Architecture
![AD-DAE Architecture](assets/Figure_architechture_diagram.png)

> Figure: High-level schematic of the AD-DAE framework showing the encoder-guided diffusion decoder, latent shift estimation, and consistency guidance.

## Inference
- A walkthrough notebook is provided: **`Inference_AD_DAE.ipynb`**  
- It demonstrates:
  - Loading a trained AD-DAE checkpoint
  - Encoding a baseline scan
  - Estimating latent shifts from progression attributes (age, cognitive status)
  - Generating follow-up images via the diffusion decoder
  - Optional evaluation/visualization steps



