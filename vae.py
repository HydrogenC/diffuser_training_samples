import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from torchmetrics import image, regression
from torch import nn
from torch import optim
import diffusers

from utils import *


def train(save_intermediate=False):
    batch_size = 48

    train_iter = load_anime_set(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = diffusers.AutoencoderKL(
        sample_size=(64, 64),
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256),
        down_block_types=(
            "DownEncoderBlock2D",
            "AttnDownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types=(
            "UpDecoderBlock2D",
            "AttnUpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        latent_channels=4
    ).to(device)

    norm_loss = nn.MSELoss().to(device)
    vgg_loss = image.LearnedPerceptualImagePatchSimilarity('vgg').to(device)
    opt = optim.AdamW(net.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.98)

    for epoch in range(50):
        batch_num = 0
        total_vgg = 0
        total_kl = 0
        total_norm = 0
        net.train()
        for X, y in tqdm(train_iter):
            X, y = map_image_to_tensor(X.to(device)), y.to(device)
            # pred_noise, = net.forward(noised_X, t, y, return_dict=False)
            dist, = net.encode(X, return_dict=False)
            recon, = net.decode(dist.sample(), return_dict=False)
            kl = dist.kl().mean()
            vgg = vgg_loss.forward(recon.clamp(-1, 1), X)
            norm = norm_loss.forward(recon, X)
            ls = (vgg + norm * 10) * 1e4 + kl

            opt.zero_grad()
            ls.backward()
            opt.step()

            total_kl += kl.item()
            total_norm += norm.item()
            total_vgg += vgg.item()
            batch_num += 1

        lr_scheduler.step()
        with torch.no_grad():
            net.eval()
            latent = torch.randn((25, 4, 16, 16), device=device)
            gen, = net.decode(latent, return_dict=False)
            gen = map_tensor_to_image(gen)
            plot_tensor(gen, 5)

            for X, y in train_iter:
                X, y = X[:12].to(device), y[:12].to(device)
                X = map_image_to_tensor(X)
                recon, = net.forward(X, return_dict=False)
                X, recon = map_tensor_to_image(X), map_tensor_to_image(recon)
                plot_tensor_list([X, recon])
                break

            total_kl /= batch_num
            total_norm /= batch_num
            total_vgg /= batch_num
            print(
                "Epoch {}, kl loss: {}, vgg loss: {}, norm loss: {}".format(epoch + 1, total_kl, total_vgg, total_norm))

        if save_intermediate and (epoch + 1) % 10 == 0:
            net.save_pretrained(f'models/anime_vae_low_{epoch + 1}_epoches')


if __name__ == '__main__':
    # train(True)

    batch_size = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net: diffusers.AutoencoderKL = diffusers.AutoencoderKL.from_pretrained('models',
                                                                           subfolder='anime_vae_l2highvgg_50_epoches').to(
        device)
    train_iter = load_anime_set(batch_size=batch_size)

    net.eval()
    with torch.no_grad():
        for X, y in train_iter:
            X, y = map_image_to_tensor(X.to(device)), y.to(device)
            latent, = net.encode(X, return_dict=False)
            print(latent.mode())
            break

        latent = torch.randn((100, 4, 16, 16), device=device)
        gen, = net.decode(latent, return_dict=False)
        plot_tensor(map_tensor_to_image(gen), 10)
