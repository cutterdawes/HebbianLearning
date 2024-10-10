
import argparse
from torch.utils.data import DataLoader
from torch import optim
from dataset import FastMNIST

# Main training loop MNIST
if __name__ == "__main__":
    # create and parse arguments
    parser = argparse.ArgumentParser(description='Train a perceptron on MNIST using specified Hebbian learning rule')
    parser.add_argument('--learning_rule', type=str, choices=['hebbs_rule', 'ojas_rule', 'hard_WTA', 'soft_WTA', 'STDP', 'random_W'], help='Choose competitive WTA rule')
    parser.add_argument('--learning_params', default='none', help='Choose optional parameters for Hebbian learning rule (default: none)')
    parser.add_argument('--n_hebbian_layers', type=int, default=1, help='Number of unsupervised Hebbian layers (default: 1)')
    parser.add_argument('--hidden_dim', type=int, default=2000, help='Number of neurons in hidden layer (default: 2000)')
    parser.add_argument('--unsup_epochs', type=int, default=1, help='Number of unsupervised epochs (default: 1)')
    parser.add_argument('--sup_epochs', type=int, default=50, help='Number of supervised epochs (default: 50)')
    parser.add_argument('--unsup_lr', type=float, default=0.001, help='Unsupervised learning rate (default: 0.001)')
    parser.add_argument('--sup_lr', type=float, default=0.001, help='Supervised learning rate (default: 0.001)')
    parser.add_argument('--unsup_batch', type=int, default=64, help='Unsupervised batch size (default: 64)')
    parser.add_argument('--sup_batch', type=int, default=64, help='Supervised batch size (default: 64)')
    parser.add_argument('--save', action='store_true', help='Save model')
    args = parser.parse_args()

    # unpack and print args
    kwargs = {}
    if args.learning_params != 'none':
        for arg in args.learning_params.split('_'):
            k, val = arg.split('=')
            if k in ['N_hebb', 'N_anti', 'K_anti']:  # NOTE: not working currently
                kwargs[k] = int(val)
            elif k == 'delta' or k == 'temp' or k == 'beta':
                kwargs[k] = float(val)
            args.learning_params = '_'.join([f'{k}={val}' for k, val in kwargs.items()])
    
    print(
        f'\nParameters:\n' + 
        f'\nlearning_rule={args.learning_rule}' +
        f'\nlearning_params={args.learning_params}' +
        f'\nn_hebbian_layers={args.n_hebbian_layers}\n' +
        f'\nhidden_dim={args.hidden_dim}' +
        f'\nunsup_epochs={args.unsup_epochs}\tsup_epochs={args.sup_epochs}' +
        f'\nunsup_lr={args.unsup_lr}\tsup_lr={args.sup_lr}' +
        f'\nunsup_batch={args.unsup_batch}\tsup_batch={args.sup_batch}'
    )

    # specify device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GenHebb(28*28, args.hidden_dim, 10, args.learning_rule, args.n_hebbian_layers, **kwargs)
    model.to(device)
    model_name = (
        f'genhebb-{args.learning_rule}-{args.learning_params}'
        f'-{args.n_hebbian_layers}_hebbian_layers-{args.hidden_dim}_hidden_dim'
        f'-{args.unsup_epochs}_unsup_epochs-{args.sup_epochs}_sup_epochs'
        f'-{args.unsup_lr}_unsup_lr-{args.sup_lr}_sup_lr'
        f'-{args.unsup_batch}_unsup_batch-{args.sup_batch}_sup_batch'
    )

    # load train and test data
    trainset = FastMNIST('./data', train=True, download=True, device=device)
    unsup_trainloader = DataLoader(trainset, batch_size=args.unsup_batch, shuffle=True)
    sup_trainloader = DataLoader(trainset, batch_size=args.sup_batch, shuffle=True)
    testset = FastMNIST('./data', train=False, download=True, device=device)
    testloader = DataLoader(testset, batch_size=args.sup_batch, shuffle=False)

    # define loss, optimizers, and LR schedulers
    criterion = nn.CrossEntropyLoss()

    # unsup_optimizer = optim.Adam(model.hebb.parameters(), lr=args.unsup_lr)
    unsup_optimizer = optim.Adam([
        {'params': model.hebb[i].parameters(), 'lr': args.unsup_lr / 10**i}  # learning rate for i-th Hebbian layer
        for i in range(model.n_hebbian_layers)
    ])
    sup_optimizer = optim.Adam(model.classifier.parameters(), lr=args.sup_lr)

    # scheduler = optim.lr_scheduler.ExponentialLR(unsup_optimizer, gamma=0.2)  # NOTE: exponential decaying lr
    # lr_lambda = lambda epoch: (-0.99 * args.unsup_lr / args.unsup_epochs) * epoch + args.unsup_lr
    # scheduler = optim.lr_scheduler.LambdaLR(unsup_optimizer, lr_lambda=lr_lambda)

    # unsupervised training with Hebbian learning rule
    print('\n\nTraining Hebbian embedding...\n')
    for epoch in range(args.unsup_epochs):
        for inputs, _ in unsup_trainloader:
            inputs = inputs.to(device)

            # zero the parameter gradients
            unsup_optimizer.zero_grad()

            # forward + update computation
            with torch.no_grad():
                outputs = model(inputs)

            # optimize
            unsup_optimizer.step()

        # scheduler.step()

        # compute Hebbian embedding statistics
        norms = [int(torch.norm(model.hebb[i].W)) for i in range(model.n_hebbian_layers)]
        norms = ', '.join(map(str, norms))
        print(f'Epoch [{epoch+1}/{args.unsup_epochs}]\t|W|_F: {norms}')

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
        for inputs, labels in sup_trainloader:
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
            if epoch == 0 or epoch % 10 == 9 or epoch == args.sup_epochs - 1:
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        # evaluation on test set
        if epoch == 0 or epoch % 10 == 9 or epoch == args.sup_epochs - 1:
            print(f'Epoch [{epoch+1}/{args.sup_epochs}]')
            print(f'train loss: {running_loss / len(sup_trainloader):.3f} \t train accuracy: {100 * correct / total:.1f} %')

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
            print(f'test loss: {running_loss / len(testloader):.3f} \t test accuracy: {100 * correct / total:.1f} % \n')

    # save model if specified
    if args.save:
        path = f'saved_models/done-training/{model_name}.pt'
        torch.save(model.state_dict(), path)
        print(f'Model saved to: {path}')