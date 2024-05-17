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
    batch_size = 20

    train_iter = load_slice_set(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = diffusers.VQModel(
        sample_size=(128, 128),
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(64, 64, 128),
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
        latent_channels=4,
        num_vq_embeddings=512
    ).to(device)

    norm_loss = nn.MSELoss().to(device)
    vgg_loss = image.LearnedPerceptualImagePatchSimilarity('vgg').to(device)
    opt = optim.AdamW(net.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.98)

    for epoch in range(50):
        batch_num = 0
        total_vgg = 0
        total_vq = 0
        total_norm = 0
        net.train()
        for i in range(10):
            for X, y in tqdm(train_iter):
                X, y = map_image_to_tensor(X.to(device)), y.to(device)
                # pred_noise, = net.forward(noised_X, t, y, return_dict=False)
                latent, = net.encode(X, return_dict=False)
                latent, vq_commit_loss, _ = net.quantize.forward(latent)
                recon, = net.decode(latent, force_not_quantize=True, return_dict=False)
                vq = vq_commit_loss.mean()
                vgg = vgg_loss.forward(recon.clamp(-1, 1), X)
                norm = norm_loss.forward(recon, X)
                ls = (vgg * 2 + norm * 10) + vq

                opt.zero_grad()
                ls.backward()
                opt.step()

                total_vq += vq.item()
                total_norm += norm.item()
                total_vgg += vgg.item()
                batch_num += 1

        lr_scheduler.step()
        with torch.no_grad():
            net.eval()
            for X, y in train_iter:
                X, y = X[:12].to(device), y[:12].to(device)
                X = map_image_to_tensor(X)
                recon, = net.forward(X, return_dict=False)
                X, recon = map_tensor_to_image(X), map_tensor_to_image(recon)
                plot_tensor_list([X, recon])
                break

            total_vq /= batch_num
            total_norm /= batch_num
            print(
                "Epoch {}, vq loss: {}, vgg loss: {}, norm loss: {}".format(epoch + 1, total_vq, total_vgg, total_norm))

        if save_intermediate and (epoch + 1) % 10 == 0:
            net.save_pretrained(f'models/slice_vqvae_{epoch + 1}_epoches')


if __name__ == '__main__':
    train(True)

    batch_size = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net: diffusers.AutoencoderKL = diffusers.AutoencoderKL.from_pretrained('models',
                                                                           subfolder='anime_vae_l2highvgg_50_epoches').to(
        device)
    train_iter = load_anime_set(batch_size=batch_size)

    net.eval()
    with torch.no_grad():
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            latent, = net.encode(X, return_dict=False)
            print(latent.mean.mean())
            print(latent.std.mean())
            break

        latent = torch.randn((100, 4, 16, 16), device=device)
        gen, = net.decode(latent, return_dict=False)
        plot_tensor(map_tensor_to_image(gen), 10)
