"""
Script to train a perceptron on MNIST using various Hebbian learning rules
"""
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FastMNIST
from learning_rules import LearningRule


class HebbianLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            plasticity: str = 'hebbs_rule',
            wta: str = 'none',
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
        self.learning_rule = LearningRule(plasticity, wta, **kwargs)
        
        # optionally normalize W
        self.normalized = normalized
        if self.normalized:
            self.W.data = F.normalize(self.W.data)

    def forward(self, x):
        # standard forward pass
        y = torch.matmul(x, self.W.T)

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
            n_hebbian_layers: int = 1,
            plasticity: str = 'hebbs_rule',
            wta: str = 'none',
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
        self.n_hebbian_layers = n_hebbian_layers
        
        # stack unsupervised Hebbian layers
        layers = []
        for i in range(n_hebbian_layers):
            if i == 0:
                layers.append(HebbianLayer(input_dim, hidden_dim, plasticity, wta, **kwargs))
            else:
                layers.append(HebbianLayer(hidden_dim, hidden_dim, plasticity, wta, **kwargs))
            layers.append(nn.ReLU())
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


# Main training loop MNIST
if __name__ == "__main__":
    # create and parse arguments
    parser = argparse.ArgumentParser(description='Train a perceptron on MNIST using specified Hebbian learning rule')
    parser.add_argument('--plasticity', type=str, default='hebbs_rule', choices=['hebbs_rule', 'ojas_rule', 'random_W'],
                        help='Choose plasticity rule (default: hebbs_rule)')
    parser.add_argument('--wta', type=str, default='none', choices=['hard', 'soft', 'none'], help='Choose competitive WTA rule (default: none)')
    parser.add_argument('--learning_params', type=str, default='none', help='Choose optional parameters for Hebbian learning rule (default: none)')
    parser.add_argument('--n_hebbian_layers', type=int, default=1, help='Number of unsupervised Hebbian layers (default: 1)')
    parser.add_argument('--hidden_dim', type=int, default=2000, help='Number of neurons in hidden layer (default: 2000)')
    parser.add_argument('--unsup_epochs', type=int, default=1, help='Number of unsupervised epochs (default: 1)')
    parser.add_argument('--sup_epochs', type=int, default=50, help='Number of supervised epochs (default: 50)')
    parser.add_argument('--unsup_lr', type=float, default=0.001, help='Unsupervised learning rate (default: 0.001)')
    parser.add_argument('--sup_lr', type=float, default=0.001, help='Supervised learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--save', action='store_true', help='Save model')
    args = parser.parse_args()

    # unpack and print args
    learning_rule = args.plasticity if args.wta == 'none' else f'{args.wta}_WTA_{args.plasticity}'
    if args.learning_params != 'none':
        kwargs = {
            k: float(val) if '.' in val else int(val)
            for k, val in
            [
                kwarg.split('=') for kwarg in
                args.learning_params.split('-')[-1].split('_')
            ]}
    else:
        kwargs = {}
    print(
        f'\nParameters:\n' + 
        f'\nlearning_rule={learning_rule}' +
        f'\nlearning_params={args.learning_params}' +
        f'\nn_hebbian_layers={args.n_hebbian_layers}' +
        f'\nhidden_dim={args.hidden_dim}\tbatch_size={args.batch_size}' +
        f'\nunsup_epochs={args.unsup_epochs}\tsup_epochs={args.sup_epochs}' +
        f'\nunsup_lr={args.unsup_lr}\tsup_lr={args.sup_lr}'
    )

    # specify device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GenHebb(28*28, args.hidden_dim, 10, args.n_hebbian_layers, args.plasticity, args.wta, **kwargs)
    model.to(device)
    model_name = (
        f'genhebb-{learning_rule}-{args.learning_params}'
        f'-{args.hidden_dim}_hidden_dim-{args.batch_size}_batch'
        f'-{args.unsup_epochs}_unsup_epochs-{args.sup_epochs}_sup_epochs'
        f'-{args.unsup_lr}_unsup_lr-{args.sup_lr}_sup_lr'
    )

    # load train and test data
    trainset = FastMNIST('./data', train=True, download=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = FastMNIST('./data', train=False, download=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # define loss, optimizers, and LR schedulers
    criterion = nn.CrossEntropyLoss()

    unsup_optimizer = optim.Adam(model.hebb.parameters(), lr=args.unsup_lr)
    sup_optimizer = optim.Adam(model.classifier.parameters(), lr=args.sup_lr)

    # unsupervised training with Hebbian learning rule
    print('\n\nTraining Hebbian embedding...\n')
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

        # compute Hebbian embedding statistics
        norms = [int(torch.norm(model.hebb[2*i].W)) for i in range(model.n_hebbian_layers)]
        norms = ', '.join(map(str, norms))
        print(f'Epoch [{epoch+1}/{args.unsup_epochs}]\t|W|_F: {norms}')
        # if args.save:
        #     path = f'saved_models/mid-training/{model_name}-epoch_{epoch+1}.pt'
        #     torch.save(model.state_dict(), path)  # NOTE: not saving mid-training

    unsup_optimizer.zero_grad()
    model.hebb.requires_grad = False
    model.hebb.eval()

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
    if args.save:
        path = f'saved_models/done-training/{model_name}.pt'
        torch.save(model.state_dict(), path)
        print(f'Model saved to: {path}')