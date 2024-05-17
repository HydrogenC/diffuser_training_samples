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
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler, PNDMScheduler
from utils import *


def train():
    batch_size = 40
    gen_steps = 500

    train_iter = load_anime_set(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scheduler = DDIMScheduler(num_train_timesteps=gen_steps)
    net = UNet2DModel(
        sample_size=(64, 64),
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        # num_class_embeds=10
    ).to(device)

    loss = nn.MSELoss().to(device)
    opt = optim.AdamW(net.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.95)

    for epoch in range(50):
        total_loss = 0
        net.train()
        for X, y in tqdm(train_iter):
            X, y = X.to(device), y.to(device)
            t = torch.randint(0, gen_steps, (X.shape[0],), device=device)
            noise = torch.randn_like(X, device=device)
            noised_X = scheduler.add_noise(X, noise, t)
            pred_noise, = net.forward(noised_X, t, return_dict=False)
            ls = loss.forward(pred_noise, noise)

            opt.zero_grad()
            ls.backward()
            opt.step()

            total_loss += ls.item()

        lr_scheduler.step()
        with torch.no_grad():
            # sample = torch.normal(0, 1, [10, 128], device=device)
            # classes = torch.nn.functional.one_hot(torch.arange(0, 10), num_classes=10)
            # classes = classes.type(torch.FloatTensor).to(device)

            net.eval()
            sample = torch.randn((10, 3, 64, 64), requires_grad=False).to(device)
            plt_tensors = []
            for t in scheduler.timesteps:
                t_batch = torch.full((10,), t, device=device)
                pred_noise, = net.forward(sample, t_batch, return_dict=False)
                sample, = scheduler.step(pred_noise, t, sample, return_dict=False)
                if t % 100 == 0:
                    plt_tensors.append(sample.detach().clone())

            plot_tensor_list(plt_tensors)
            del plt_tensors
            print("Epoch {}, train loss: {}".format(epoch, total_loss))

    net.save_pretrained('models/anime')


if __name__ == '__main__':
    gen_steps = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scheduler = DDPMScheduler(num_train_timesteps=gen_steps)
    # scheduler = PNDMScheduler(num_train_timesteps=gen_steps)
    net = UNet2DModel.from_pretrained('models', subfolder='anime').to(device)

    # scheduler.set_timesteps(100, device)
    net.eval()
    with torch.no_grad():
        sample = torch.randn((10, 3, 64, 64), requires_grad=False).to(device)
        plt_tensors = []
        for t in scheduler.timesteps:
            t_batch = torch.full((10,), t, device=device)
            pred_noise, = net.forward(sample, t_batch, return_dict=False)
            sample, = scheduler.step(pred_noise, t, sample, return_dict=False)
            if t % 100 == 0 or t <= 1:
                plt_tensors.append(sample.detach().clone())

        plot_tensor_list(plt_tensors)
