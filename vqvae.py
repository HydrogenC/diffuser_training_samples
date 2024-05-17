import numpy as np
import torch
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils import data
from torch import nn
from torchmetrics import image
from torch import optim
import diffusers
from utils import *


def train(save_intermediate=False):
    batch_size = 48
    cb_count = 512

    train_iter = load_anime_set(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = diffusers.VQModel(
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
        latent_channels=4,
        num_vq_embeddings=cb_count
    ).to(device)

    norm_loss = nn.MSELoss().to(device)
    vgg_loss = image.LearnedPerceptualImagePatchSimilarity('vgg').to(device)
    opt = optim.AdamW(net.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.98)

    for epoch in range(50):
        batch_num = 0
        total_vgg = 0
        total_vq = 0
        total_norm = 0
        net.train()
        for X, y in tqdm(train_iter):
            X, y = map_image_to_tensor(X.to(device)), y.to(device)
            latent, = net.encode(X, return_dict=False)
            latent, vq_commit_loss, _ = net.quantize.forward(latent)
            recon, = net.decode(latent, force_not_quantize=True, return_dict=False)
            vq = vq_commit_loss.mean()
            vgg = vgg_loss.forward(recon.clamp(-1, 1), X)
            norm = norm_loss.forward(recon, X)
            ls = (vgg + norm * 10) * 10 + vq

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
            total_vgg /= batch_num
            print(
                "Epoch {}, vq loss: {}, vgg loss: {}, norm loss: {}".format(epoch + 1, total_vq, total_vgg, total_norm))

        if save_intermediate and (epoch + 1) % 10 == 0:
            net.save_pretrained(f'models/anime_vqvae_highrec_{epoch + 1}_epoches')


if __name__ == '__main__':
    # train(True)

    batch_size = 12
    cb_count = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net: diffusers.VQModel = diffusers.VQModel.from_pretrained('models', subfolder='anime_vqvae_low_50_epoches').to(
        device)

    train_iter = load_anime_set(batch_size)

    net.eval()
    with torch.no_grad():
        for X, y in train_iter:
            X, y = map_image_to_tensor(X.to(device)), y.to(device)
            recon, = net.forward(X, return_dict=False)
            X, recon = map_tensor_to_image(X), map_tensor_to_image(recon)
            plot_tensor_list([X, recon])
            break

        indexes = torch.randint(0, cb_count, (25, 16, 16), device=device)
        latents = net.quantize.get_codebook_entry(indexes, (25, 16, 16, 4))
        gen, = net.decode(latents, force_not_quantize=True, return_dict=False)
        gen = map_tensor_to_image(gen)
        plot_tensor(gen, 5)
