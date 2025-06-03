"""
Script to train a perceptron on MNIST using various Hebbian learning rules
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.learning_rules import LearningRule
from models.activations import Activation


class HebbianLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            learning_rule: str,
            activation: str,  # NOTE: experiment w/ ReLU, tanh, softmax
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
        self.a = Activation(activation)
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
            activation: str = 'relu',
            n_hebbian_layers: int = 1,
            importance_factor: float = 0,  # TODO: better name
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
        self.activation = Activation(activation)
        self.n_hebbian_layers = n_hebbian_layers
        self.importance_factor = importance_factor
        
        # stack unsupervised Hebbian layers
        layers = []
        for l in range(n_hebbian_layers):
            if l == 0:
                layers.append(HebbianLayer(input_dim, hidden_dim, learning_rule, activation, **kwargs))
            else:
                layers.append(HebbianLayer(hidden_dim, hidden_dim, learning_rule, activation, **kwargs))
        self.hebb = nn.Sequential(*layers)

        # add classifier layer
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 28*28)  # NOTE: specific to MNIST
        # unsupervised Hebbian embedding
        x = self.hebb(x)
        # linear classifier
        y = self.classifier(x)
 
        # modulate by relative importance of neuron (if not last Hebbian layer)
        # TODO: clean up implementation
        if self.importance_factor != 0 and self.training:
            for l in range(self.n_hebbian_layers):
                if l < self.n_hebbian_layers - 1:
                    W_nl = self.hebb[l+1].W
                    imp_l = torch.norm(W_nl, dim=0).unsqueeze(-1)
                    # self.hebb[l].W.grad /= imp_l**self.importance_factor
                    
                    # imp_l = torch.where(imp_l >= 1, imp_l, -imp_l)
                    imp_l += torch.ones_like(imp_l)
                    self.hebb[l].W.grad *= imp_l

        return y