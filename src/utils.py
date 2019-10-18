import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import List
import numpy as np


def std_normalize_samplewise_ndarray(data_sample: np.ndarray) -> np.ndarray:
    dims = tuple(range(1, len(data_sample.shape)))
    means = np.mean(data_sample, axis=dims, keepdims=True)
    std = np.std(data_sample, axis=dims, keepdims=True)
    return (data_sample - means)/std


class SpectralWrapper:
    def __init__(self, use_sn=True):
        self.use_sn = use_sn

    def wrap(self, module: torch.nn.Module, **kwargs):
        if self.use_sn:
            print("wrapping ", module._get_name())
            return spectral_norm(module, **kwargs)
        else:
            return module


def gradient_penalty(true, fake, model, k=2, p=6):
    batch_size, *other_dims = true.size()
    mu = torch.rand(batch_size, requires_grad=True)
    for i in range(len(other_dims)):
        mu = mu.unsqueeze(-1)
    mu = mu.expand(-1, *other_dims)
    avg_sample = (1 + (-mu)) * true + mu * fake
    avg_sample.retain_grad()

    validity = model(avg_sample)
    grad, = torch.autograd.grad(outputs=validity,
                                inputs=avg_sample,
                                grad_outputs=torch.ones_like(validity),
                                only_inputs=True,
                                create_graph=True,
                                retain_graph=True)
    grad_norm = torch.norm(grad, dim=-1, keepdim=True)
    grad_penalty = (k * torch.pow(grad_norm, p)).mean()
    return grad_penalty


def to_tensors(data_samples, device: torch.device = None) -> List[torch.Tensor]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_list = []
    for sample in data_samples:
        tensor_list.append(torch.from_numpy(sample).float().to(device))
    return tensor_list


def mesh(start, finish, steps):
    x = np.linspace(start, finish, steps)
    y = np.linspace(start, finish, steps)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
