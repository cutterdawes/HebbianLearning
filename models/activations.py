"""
Script containing general activation class
"""
import torch
from torch import nn


class Activation:
    def __init__(
            self,
            activation: str
    ):
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh
        }
        if activation not in activations.keys():
            raise ValueError(f'Argument activation={activation} must be among {activations.keys()}')
        self.activation = activations[activation]()

    def __call__(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.activation(x)