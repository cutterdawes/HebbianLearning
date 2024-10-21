import torch
from torch import nn, optim
from models.baseline import Baseline

import logging


def unsupervised(model, trainloader, epochs, lr, device):
    '''Unsupervised training with Hebbian learning rule'''

    # log start of training
    logging.info('\nunsupervised training...\n')

    # Prepare model
    model.to(device)
    model.hebb.train()

    # define unsupervised optimizer and LR scheduler
    # optimizer = optim.Adeam(model.hebb.parameters(), lr=lr)
    optimizer = optim.Adam([
        {'params': model.hebb[i].parameters(), 'lr': lr / 2**i}  # learning rate for i-th Hebbian layer
        for i in range(model.n_hebbian_layers)
    ])

    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2)  # NOTE: exponential decaying lr
    # lr_lambda = lambda epoch: (-0.99 * lr / epochs) * epoch + lr
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Unsupervised training loop
    for epoch in range(epochs):
        for inputs, _ in trainloader:
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + update computation
            with torch.no_grad():
                outputs = model(inputs)

            # optimize
            optimizer.step()

        # scheduler.step()

        # compute Hebbian embedding statistics
        norms = [int(torch.norm(model.hebb[i].W)) for i in range(model.n_hebbian_layers)]
        norms = ', '.join(map(str, norms))
        msg = f'epoch [{epoch+1}/{epochs}]\t|W|_F: {norms}'
        logging.info(msg)

    optimizer.zero_grad()
    model.hebb.requires_grad = False
    model.hebb.eval()


def supervised(model, trainloader, testloader, epochs, lr, device):
    '''supervised training of classifier'''

    # log start of training
    logging.info('\n\nsupervised training...\n')

    # Prepare model
    model.to(device)
    if isinstance(model, Baseline):
        model.train()

    # define loss and optimizer
    if isinstance(model, Baseline):
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Supervised training loop
    for epoch in range(epochs):
        model.classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute training statistics
            running_loss += loss.item()
            if epoch == 0 or epoch % 10 == 9 or epoch == epochs - 1:
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        # evaluation on test set
        if epoch == 0 or epoch % 10 == 9 or epoch == epochs - 1:
            msg = (
                f'epoch [{epoch+1}/{epochs}]\n' +
                f'train loss: {running_loss / len(trainloader):.3f} \t train accuracy: {100 * correct / total:.1f} %'
            )
            logging.info(msg)

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
            msg = f'test loss: {running_loss / len(testloader):.3f} \t test accuracy: {100 * correct / total:.1f} % \n'
            logging.info(msg)