import diffusers
import numpy as np
import torch
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import ImageFolder
from torch.utils import data
from torch import nn
from torch import optim
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from utils import *

# VQ-VAE or KL-based VAE
use_kl_model = False


def train(save_intermediate=False):
    batch_size = 100
    gen_steps = 500

    train_iter = load_anime_set(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_kl_model:
        scheduler = DDPMScheduler(num_train_timesteps=gen_steps, clip_sample_range=2)
        vae: diffusers.AutoencoderKL = diffusers.AutoencoderKL.from_pretrained('models',
                                                                               subfolder='anime_vae_l2highvgg_50_epoches').to(
            device)
    else:
        scheduler = DDPMScheduler(num_train_timesteps=gen_steps, clip_sample_range=8)
        vae: diffusers.VQModel = diffusers.VQModel.from_pretrained('models', subfolder='anime_vqvae_low_50_epoches').to(
            device)

    unet = UNet2DModel(
        sample_size=(16, 16),
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        # num_class_embeds=10
    ).to(device)

    loss = nn.MSELoss().to(device)
    lr = 1e-3 if use_kl_model else 1e-4
    opt = optim.AdamW(unet.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.98)

    vae.eval()
    for epoch in range(50):
        total_loss = 0
        unet.train()
        for X, y in tqdm(train_iter):
            X, y = map_image_to_tensor(X.to(device)), y.to(device)
            latent, = vae.encode(X, return_dict=False)
            if use_kl_model:
                latent = latent.mode().detach()
            else:
                latent, _, _ = vae.quantize.forward(latent)
                latent = latent.detach()

            t = torch.randint(0, gen_steps, (X.shape[0],), device=device)
            noise = torch.randn_like(latent, device=device)
            noised_latent = scheduler.add_noise(latent, noise, t)
            pred_noise, = unet.forward(noised_latent, t, return_dict=False)
            ls = loss.forward(pred_noise, noise)

            opt.zero_grad()
            ls.backward()
            opt.step()

            total_loss += ls.item()

        lr_scheduler.step()
        with torch.no_grad():
            unet.eval()
            sample = torch.randn((10, 4, 16, 16), requires_grad=False).to(device)
            plt_tensors = []
            for t in scheduler.timesteps:
                pred_noise, = unet.forward(sample, t, return_dict=False)
                sample, = scheduler.step(pred_noise, t, sample, return_dict=False)
                if t % 100 == 0:
                    img, = vae.decode(sample, return_dict=False)
                    plt_tensors.append(map_tensor_to_image(img.detach()))

            plot_tensor_list(plt_tensors)
            del plt_tensors
            print("Epoch {}, train loss: {}".format(epoch + 1, total_loss))

        if save_intermediate and (epoch + 1) % 10 == 0:
            unet.save_pretrained(f'models/anime_latent_vq_{epoch + 1}_epoches')


if __name__ == '__main__':
    train(True)

    gen_steps = 500
    train_iter = load_anime_set(batch_size=10)
    device = cuda_if_possible()

    if use_kl_model:
        scheduler = DDPMScheduler(num_train_timesteps=gen_steps, clip_sample_range=2)
    else:
        scheduler = DDPMScheduler(num_train_timesteps=gen_steps, clip_sample_range=8)
    # scheduler = PNDMScheduler(num_train_timesteps=gen_steps)
    vae: diffusers.VQModel = diffusers.VQModel.from_pretrained('models', subfolder='anime_vqvae_low_50_epoches').to(
        device)
    net = UNet2DModel.from_pretrained('models', subfolder='anime_latent_vq_50_epoches').to(device)

    # scheduler.set_timesteps(100, device)
    vae.eval()
    net.eval()
    with torch.no_grad():
        sample = torch.randn((10, 4, 16, 16), requires_grad=False).to(device)
        plt_tensors = []
        for t in scheduler.timesteps:
            pred_noise, = net.forward(sample, t, return_dict=False)
            sample, = scheduler.step(pred_noise, t, sample, return_dict=False)
            if t % 100 == 0 or t <= 1:
                img, = vae.decode(sample, return_dict=False)
                plt_tensors.append(map_tensor_to_image(img.detach()))

        plot_tensor_list(plt_tensors)
