"""
Script that provides a baseline trained with BP end-to-end to compare with various Hebbian learning rules
"""
from torch import nn


class Baseline(nn.Module):
    """
    Simple baseline MLP to be trained with BP end-to-end
    """
    def __init__(self,
                input_dim: int,
                hidden_dim: int,
                output_dim: int,
                n_hidden_layers: int = 1,
                dropout_p: float = 0
                ) -> None:
        super(Baseline, self).__init__()

        # set model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # stack hidden layers
        layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
        self.hidden = nn.Sequential(*layers)

        # add classifier layer
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  #  NOTE: specific to MNIST
        x = self.hidden(x)
        y = self.classifier(x)
        return y