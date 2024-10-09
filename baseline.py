"""
Script that provides a baseline trained with BP end-to-end to compare with various Hebbian learning rules
"""
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import FastMNIST


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


# Main training loop MNIST
if __name__ == "__main__":
    # create and parse arguments
    parser = argparse.ArgumentParser(description='Train a perceptron on MNIST with BP end-to-end')
    parser.add_argument('--hidden_dim', type=int, default=2000, help='Number of neurons in hidden layer (default: 2000)')
    parser.add_argument('--n_hidden_layers', type=int, default=1, help='Number of hidden layers (default: 1)')
    parser.add_argument('--dropout_p', type=float, default=0, help='Dropout probability (default: 0)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--save', action='store_true', help='Save the model')
    args = parser.parse_args()

    print(
        f'\nParameters:\n' + 
        f'\nlearning_rule=supervised_BP' +
        f'\nn_hidden_layers={args.n_hidden_layers}' +
        f'\ndropout_p={args.dropout_p}\n' +
        f'\nhidden_dim={args.hidden_dim}' +
        f'\nepochs={args.epochs}' + 
        f'\nlr={args.learning_rate}' +
        f'\nbatch={args.batch}'
    )

    # specify device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Baseline(28*28, args.hidden_dim, 10, args.n_hidden_layers, args.dropout_p)
    model.to(device)
    model_name = f'baseline-{args.n_hidden_layers}_hidden_layers-{args.dropout_p}_dropout_p-{args.epochs}_epochs-{args.learning_rate}_lr-{args.batch}_batch'

    # load train and test data
    trainset = FastMNIST('./data', train=True, download=True)
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)

    testset = FastMNIST('./data', train=False, download=True)
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # train baseline model (BP end-to-end)
    print('\n\nTraining...\n')
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
        path = f'saved_models/done-training/{model_name}.pt'
        torch.save(model.state_dict(), path)
        print(f'Model saved to: {path}')