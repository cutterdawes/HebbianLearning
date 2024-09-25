"""
Script to train a perceptron on MNIST using various Hebbian learning rules
"""
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from typing import Callable


def hebbs_rule(x, y, W):
    """
    return dW according to Hebb's rule: dW = y x^T
    """
    dW = y.unsqueeze(-1) * x.unsqueeze(-2)
    if dW.dim() > 2:
        dW = torch.mean(dW, 0)
    return dW

def ojas_rule(x, y, W):
    """
    return dW according to Oja's rule: dW_ij = y_i (x_j - y_i W_ij)
    """
    dW = y.unsqueeze(-1) * x.unsqueeze(-2) - (y**2).unsqueeze(-1) * W.unsqueeze(0)
    if dW.dim() > 2:
        dW = torch.mean(dW, 0)
    return dW

def hard_WTA(learning_rule):
    """
    decorator to add hard WTA to specified learning rule (i.e., only change weights of "winning" neuron)
    """
    def hard_WTA_learning_rule(x, y, W):

        # find winning neuron and create indicator tensor
        ind_win = torch.zeros_like(y)
        winners = torch.argmax(y, -1)
        ind_win = F.one_hot(winners, y.shape[-1]).float()

        # modify dW to only change weights of winning neuron
        dW = learning_rule(x, y, W)
        dW = ind_win.unsqueeze(-1) * dW.unsqueeze(0)
        if dW.dim() > 2:
            dW = torch.mean(dW, 0)

        return dW
    
    return hard_WTA_learning_rule

@hard_WTA
def hard_WTA_hebbs_rule(x, y, W):
    """
    hard WTA added to Hebb's rule
    """
    return hebbs_rule(x, y, W)

@hard_WTA
def hard_WTA_ojas_rule(x, y, W):
    """
    hard WTA added to Oja's rule
    """
    return ojas_rule(x, y, W)

def random_W(x, y, W):
    """
    return dW = 0 so that weights remain at random initialization (alt. baseline test)
    """
    dW = torch.zeros_like(W)
    return dW


learning_rules = {
    'hebbs_rule': hebbs_rule,
    'ojas_rule': ojas_rule,
    'hard_WTA_hebbs_rule': hard_WTA_hebbs_rule,
    'hard_WTA_ojas_rule': hard_WTA_ojas_rule,
    'random_W': random_W
}


class HebbianLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            learning_rule: Callable[
                [torch.Tensor, torch.Tensor, nn.Parameter],
                torch.Tensor
            ],
            normalized: bool = True  # NOTE: add to script args?
    ) -> None:
        """
        Fully-connected layer that updates via Hebb's rule
        """
        super(HebbianLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn(output_dim, input_dim))
        self.learning_rule = learning_rule
        self.normalized = normalized
        if self.normalized:
            self.W.data = F.normalize(self.W.data)

    def forward(self, x):
        # standard forward pass
        y = torch.matmul(x, self.W.T)

        # compute specified Hebbian learning rule, store in grad
        if self.training:
            dW = self.learning_rule(x, y, self.W)
            self.W.grad = -dW

        return y
    

class GenHebb(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            learning_rule: Callable[
                [torch.Tensor, torch.Tensor, nn.Parameter],
                torch.Tensor
            ]
    ) -> None:
        """
        One-layer fully-connected model with a very simple Hebbian learning rule
        """
        super(GenHebb, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.unsup_layer = HebbianLayer(input_dim, hidden_dim, learning_rule)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 28*28)  # NOTE: specific to MNIST
        # unsupervised Hebbian layer
        x = F.relu(self.unsup_layer(x))
        # linear classifier
        y = self.classifier(x)
        return y
        
    
class FastMNIST(MNIST):
    """
    Improves performance of training on MNIST by removing the PIL interface and pre-loading on the GPU (2-3x speedup)

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', 'cpu')
        train_class = kwargs.pop('train_class', 'all')
        split = kwargs.pop('split', 'train')
        super().__init__(*args, **kwargs)

        self.split = split

        if self.train:
            if not isinstance(train_class, str):
                print(train_class)
                self.targets = np.array(self.targets)
                index_class = np.isin(self.targets, train_class)
                self.data = self.data[index_class]
                self.targets = self.targets[index_class]
                self.len = self.data.shape[0]

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.to(device)

        self.targets = self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


# Main training loop MNIST
if __name__ == "__main__":
    # create and parse arguments
    parser = argparse.ArgumentParser(description='Train a perceptron on MNIST using specified Hebbian learning rule')
    parser.add_argument('--learning_rule', type=str, default='hebbs_rule', choices=learning_rules.keys(),
                        help='Choose Hebbian learning rule')
    parser.add_argument('--hidden_dim', type=int, default=2000, help='Number of neurons in hidden layer (default: 2000)')
    parser.add_argument('--unsup_epochs', type=int, default=1, help='Number of unsupervised epochs (default: 1)')
    parser.add_argument('--sup_epochs', type=int, default=50, help='Number of supervised epochs (default: 50)')
    parser.add_argument('--unsup_lr', type=float, default=0.001, help='Unsupervised learning rate (default: 0.001)')
    parser.add_argument('--sup_lr', type=float, default=0.001, help='Supervised learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--save', action='store_true', help='Save model')
    args = parser.parse_args()

    print(
        f'\nParameters:\n' + 
        f'\nlearning_rule={args.learning_rule}' +
        f'\nhidden_dim={args.hidden_dim}\tbatch_size={args.batch_size}' +
        f'\nunsup_epochs={args.unsup_epochs}\tsup_epochs={args.sup_epochs}' +
        f'\nunsup_lr={args.unsup_lr}\tsup_lr={args.sup_lr}'
    )

    # specify device, learning rule, and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rule = learning_rules[args.learning_rule]
    model = GenHebb(28*28, args.hidden_dim, 10, learning_rule)
    model.to(device)
    model_name = (
        f'genhebb-{args.learning_rule}'
        f'-{args.unsup_epochs}_unsup_epochs-{args.sup_epochs}_sup_epochs'
        f'-{args.unsup_lr}_unsup_lr-{args.sup_lr}_sup_lr-{args.batch_size}_batch'
    )

    # load train and test data
    trainset = FastMNIST('./data', train=True, download=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = FastMNIST('./data', train=False, download=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # define loss, optimizers, and LR schedulers
    criterion = nn.CrossEntropyLoss()

    unsup_optimizer = optim.Adam(model.unsup_layer.parameters(), lr=args.unsup_lr)
    sup_optimizer = optim.Adam(model.classifier.parameters(), lr=args.sup_lr)

    # unsup_scheduler = ExponentialLR(unsup_optimizer, gamma=0.8)  # NOTE: scheduler is turned off

    # unsupervised training with Hebbian learning rule
    print('\n\nTraining unsupervised layer...\n')
    for epoch in range(args.unsup_epochs):
        for inputs, _ in trainloader:
            inputs = inputs.to(device)

            # zero the parameter gradients
            unsup_optimizer.zero_grad()

            # forward + update computation
            with torch.no_grad():
                outputs = model(inputs)

            # optimize
            unsup_optimizer.step()
        # unsup_scheduler.step()  # NOTE: scheduler is turned off

        # compute unsupervised layer statistics
        print(f'Epoch [{epoch+1}/{args.unsup_epochs}]\t|W|_F: {int(torch.norm(model.unsup_layer.W))}')
        if args.save:
            path = f'saved_models/mid-training/{model_name}-epoch_{epoch+1}.pt'
            torch.save(model.state_dict(), path)

    unsup_optimizer.zero_grad()
    model.unsup_layer.requires_grad = False
    model.unsup_layer.eval()

    # supervised training of classifier
    print('\n\nTraining supervised classifier...\n')
    for epoch in range(args.sup_epochs):
        model.classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            sup_optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            sup_optimizer.step()

            # compute training statistics
            running_loss += loss.item()
            if epoch % 10 == 0 or epoch == 49:
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        # evaluation on test set
        if epoch % 10 == 0 or epoch == 49:
            print(f'Epoch [{epoch+1}/{args.sup_epochs}]')
            print(f'train loss: {running_loss / len(trainloader):.3f} \t train accuracy: {100 * correct / total:.1f} %')

            # on the test set
            model.eval()
            running_loss = 0.
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # calculate outputs by running inputs through the network
                    outputs = model(inputs)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
            print(f'test loss: {running_loss / len(trainloader):.3f} \t test accuracy: {100 * correct / total:.1f} % \n')

    # save model if specified
    import pdb; pdb.set_trace()
    if args.save:
        path = f'saved_models/done-training/{model_name}.pt'
        torch.save(model.state_dict(), path)
        print(f'Model saved to: {path}')