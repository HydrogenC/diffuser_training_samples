# Diffuser train samples
A series of samples demonstrating how to train simple samples with the lib `diffusers`.  
May be a lot easier to read for beginners since the complicated conditioning stuff (e.g. CLIP) is removed. 
Can be a beginner's guide for the library `diffusers`. 
- `vae.py` is a simple (unconditional) variational autoencoder
- `vqvae.py` is a simple (unconditional) vector-quantized VAE
- `pixel_diffusion.py` is a (unconditional) diffusion model that can generates images out of noise directly
- `latent_diffusion.py` is a (unconditional) latent diffusion model that generates latent vector, can work with 
  either `vae.py` or `vqvae.py` to produce a final image