"""
Script to train a perceptron on MNIST using various Hebbian learning rules
"""
import torch
from torch import nn
import torch.nn.functional as F
from learning_rules import LearningRule


class HebbianLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            learning_rule: LearningRule,
            normalized: bool = True,  # NOTE: add to script args?
            **kwargs  # optional learning rule parameters
    ) -> None:
        """
        Fully-connected layer that updates via Hebb's rule
        """
        super(HebbianLayer, self).__init__()

        # set model parameters and learning rule
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn(output_dim, input_dim))
        self.a = nn.ReLU()  # NOTE: experiment w/ ReLU, tanh, softmax
        self.learning_rule = LearningRule(learning_rule, **kwargs)
        
        # optionally normalize W
        self.normalized = normalized
        if self.normalized:
            self.W.data = F.normalize(self.W.data)

    def forward(self, x):
        # standard forward pass
        y = self.a(torch.matmul(x, self.W.T))

        # compute specified Hebbian learning rule, store in grad
        if self.training:
            dW = self.learning_rule(x, y, self.W)
            self.W.grad = -dW  # negate bc gradient descent

        return y
    

class GenHebb(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            learning_rule: str,
            n_hebbian_layers: int = 1,
            **kwargs  # optional learning rule parameters
    ) -> None:
        """
        Multi-layer fully-connected model with a very simple Hebbian learning rule, topped by one-layer linear classifier
        """
        super(GenHebb, self).__init__()

        # set model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rule = LearningRule(learning_rule, **kwargs)
        self.n_hebbian_layers = n_hebbian_layers
        
        # stack unsupervised Hebbian layers
        layers = []
        for i in range(n_hebbian_layers):
            if i == 0:
                layers.append(HebbianLayer(input_dim, hidden_dim, learning_rule, **kwargs))
            else:
                layers.append(HebbianLayer(hidden_dim, hidden_dim, learning_rule, **kwargs))
        self.hebb = nn.Sequential(*layers)

        # add classifier layer
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 28*28)  # NOTE: specific to MNIST
        # unsupervised Hebbian embedding
        x = self.hebb(x)
        # linear classifier
        y = self.classifier(x)
        return y