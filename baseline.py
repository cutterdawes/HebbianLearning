"""
Script that provides a baseline trained with BP end-to-end to compare with various Hebbian learning rules
"""
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from genhebb import FastMNIST


class Baseline(nn.Module):
    """
    Simple baseline MLP to be trained with BP end-to-end
    """
    def __init__(self):
        super(Baseline, self).__init__()
        self.input = nn.Linear(28*28, 2000)
        self.output = nn.Linear(2000, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.input(x))
        x = self.output(x)
        return x


# Main training loop MNIST
if __name__ == "__main__":
    # create and parse arguments
    parser = argparse.ArgumentParser(description='Train a perceptron on MNIST using specified Hebbian plasticity rule')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('-s', '--save', action='store_true', help='Save the model')
    args = parser.parse_args()

    # specify device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Baseline()
    model.to(device)

    # load train and test data
    trainset = FastMNIST('./data', train=True, download=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = FastMNIST('./data', train=False, download=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # train baseline model (BP end-to-end)
    print('Training...')
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute training statistics
            running_loss += loss.item()
            if epoch % 10 == 0 or epoch == 49:
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        # Evaluation on test set
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
        torch.save(model.state_dict(), 'saved_models/baseline.pt')