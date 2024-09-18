"""
Script to train a perceptron on MNIST using various Hebbian learning rules
"""
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from typing import Callable


def hebbs_rule(x, y, W):
    """
    return dW according to Hebb's rule (dW = x * y^T)
    """
    dW = x.unsqueeze(-1) * y.unsqueeze(-2)
    if dW.dim() > 2:
        dW = torch.sum(dW, 0)
    return dW


class HebbianLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            learning_rule: Callable[
                [torch.Tensor, torch.Tensor, nn.Parameter],
                torch.Tensor
            ]
    ) -> None:
        """
        One fully-connected layer that updates via Hebb's rule
        """
        super(HebbianLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        self.learning_rule = learning_rule

    def forward(self, x):
        # standard forward pass
        y = torch.matmul(x, self.W)

        # compute specified Hebbian learning rule, store in grad
        if self.training:
            dW = self.learning_rule(x, y, self.W)
            self.W.grad = dW

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
        x = x.view(-1, 28*28)
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
        device = kwargs.pop('device', "cpu")
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
        # self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255).unsqueeze(1)

        self.targets = self.targets.to(device)
        # self.targets = torch.tensor(self.targets, device=device)

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
    parser = argparse.ArgumentParser(description='Train a perceptron on MNIST using specified Hebbian plasticity rule')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('-s', '--save', action='store_true', help='Save the model')
    args = parser.parse_args()

    # specify device, model, and learning rule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenHebb(28*28, 2000, 10, hebbs_rule)
    model.to(device)

    # load train and test data
    trainset = FastMNIST('./data', train=True, download=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = FastMNIST('./data', train=False, download=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # define loss and optimizers
    criterion = nn.CrossEntropyLoss()
    unsup_optimizer = optim.Adam(model.unsup_layer.parameters(), lr=args.learning_rate)
    sup_optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # unsupervised training with Hebbian learning rule
    print('Training unsupervised layer...')
    for inputs, _ in trainloader:
        inputs = inputs.to(device)

        # zero the parameter gradients
        unsup_optimizer.zero_grad()

        # forward + update computation
        with torch.no_grad():
            outputs = model(inputs)

        # optimize
        unsup_optimizer.step()

    unsup_optimizer.zero_grad()
    model.unsup_layer.requires_grad = False
    model.unsup_layer.eval()

    # supervised training of classifier
    print('Training supervised classifier...')
    for epoch in range(args.epochs):
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
            print(f'Epoch [{epoch+1}/{args.epochs}]')
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
    if args.save:
        torch.save(model.state_dict(), f'saved_models/genhebb-hebbs_rule-{args.epochs}epochs-lr{args.learning_rate}-batch{args.batch_size}.pt')