import math

from torchvision.datasets import ImageFolder
from torch.utils import data
from torchvision.transforms import v2
import torch
import matplotlib.pyplot as plt
from torch import nn


def do_classifier_free_guidance(sample_cond: torch.Tensor, sample_uncond: torch.Tensor, cfg_scale: float):
    return sample_uncond + (sample_cond - sample_uncond) * cfg_scale


def cuda_if_possible():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def map_image_to_tensor(image: torch.Tensor) -> torch.Tensor:
    return image * 2. - 1.


def map_tensor_to_image(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp((tensor + 1.) * .5, 0, 1)


def load_anime_set(batch_size: int):
    train_augs = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    anime_set = ImageFolder(
        root='data/anime',
        transform=train_augs
    )

    train_iter = data.DataLoader(anime_set, batch_size, True, num_workers=4)
    return train_iter


def load_slice_set(batch_size: int):
    train_augs = v2.Compose([
        v2.RandomCrop((128, 128)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    anime_set = ImageFolder(
        root='data/slices',
        transform=train_augs,
    )

    train_iter = data.DataLoader(anime_set, batch_size, True, num_workers=16)
    return train_iter


def plot_tensor(tensor: torch.Tensor, cols: int, permute=True, save_name: str = None):
    rows = int(math.ceil(tensor.shape[0] / cols))
    fig = plt.figure(figsize=(cols, rows))
    tensor = tensor.detach().cpu().squeeze()
    if permute:
        tensor = tensor.permute(0, 2, 3, 1)
    num_graphs = tensor.shape[0]
    for i in range(num_graphs):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(tensor[i].clamp(0, 1), cmap='gray')

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)
    plt.close()


def plot_tensor_list(tensors: list[torch.Tensor], permute=True, save_name: str = None):
    cols = len(tensors)
    rows = tensors[0].shape[0]
    fig = plt.figure(figsize=(cols, rows))
    tensors = list(map(lambda x: x.detach().cpu().squeeze(), tensors))
    if permute:
        tensors = list(map(lambda x: x.permute(0, 2, 3, 1), tensors))
    plt_index = 1
    for i in range(rows):
        for tensor in tensors:
            fig.add_subplot(rows, cols, plt_index)
            plt.imshow(tensor[i].clamp(0, 1), cmap='gray')
            plt_index += 1

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)
    plt.close()
