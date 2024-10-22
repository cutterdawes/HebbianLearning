import torch
from torch import nn


class Model(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            n_hebbian_layers: int
    ) -> None:
        super(Model, self).__init__()

        # set model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_hebbian_layers = n_hebbian_layers

        # stack unsupervised Hebbian layers
        layers = []
        for i in range(n_hebbian_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.ReLU())
        self.hebb = nn.Sequential(*layers)

        # add classifier layer
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 28*28)  # NOTE: specific to MNIST
        # unsupervised Hebbian embedding
        y = self.hebb(x)
        # linear classifier
        y = self.classifier(y)

        # compute learning rule for each layer, store in grad
        if self.training:
            for l in range(self.n_hebbian_layers):
                # set layer-specific variables
                W_l = self.hebb[2*l].weight
                x_l = self.hebb[:2*l-1](x) if l > 0 else x
                y_l = self.hebb[:2*l+1](x)

                # compute Oja's rule
                dW_l = y_l.unsqueeze(-1) * (x_l.unsqueeze(-2) - y_l.unsqueeze(-1) * W_l.unsqueeze(0))

                # modulate by relative importance of neuron (if not last Hebbian layer)
                if l < self.n_hebbian_layers - 1:
                    W_nl = self.hebb[2*(l+1)].weight
                    imp_l = torch.norm(W_nl, dim=0).unsqueeze(-1).expand_as(dW_l)
                    dW_l /= imp_l

                # take mean if computed over batch
                if dW_l.dim() > 2:
                    dW_l = torch.mean(dW_l, 0)

                # set grad
                self.hebb[2*l].weight.grad = -dW_l # negate bc gradient descent
        
        return y