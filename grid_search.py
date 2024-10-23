import logging
import itertools
import yaml

import training
import utils


# setup logging
utils.setup_logging(log_file='grid_search.log')
logging.info('\nSTDP hyperparameter grid...\n')

# hyperparameters to search
param_grid = {
    'activation': ['relu', 'tanh'],
    'beta': [0, 0.1, 0.2, 0.5, 1],
    'epochs': [1, 3, 5, 10, 20],
    'lr': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
}
msg = yaml.dump(param_grid, sort_keys=False, default_flow_style=False)
logging.info(msg)

# grid search
logging.info('\nSearching...\n')
for activation, beta, epochs, lr in itertools.product(
        param_grid['activation'],
        param_grid['beta'],
        param_grid['epochs'],
        param_grid['lr']
    ):
    # create config
    config = {
        'model': {
            'type': 'GenHebb',
            'input_dim': 784,
            'hidden_dim': 100,
            'output_dim': 10,
            'learning_rule': 'STDP',
            'activation': activation,
            'n_hebbian_layers': 1,
            'importance_factor': 0,
            'kwargs': {'beta': beta}
        },
        'training': {
            'unsupervised': {
                'epochs': epochs,
                'lr': lr,
                'batch_size': 128
            },
            'supervised': {
                'epochs': 50,
                'lr': 0.001,
                'batch_size': 64
            }
        }
    }

    # get device
    device = utils.get_device()

    # load model
    model = utils.get_model(config['model'])

    # unsupervised training
    unsup_trainloader = utils.load_data(config['training'], 'unsupervised', device)
    training.unsupervised(
        model,
        unsup_trainloader,
        config['training']['unsupervised']['epochs'],
        config['training']['unsupervised']['lr'],
        device,
        verbose=False
    )

    # supervised training
    sup_trainloader, sup_testloader = utils.load_data(config['training'], 'supervised', device)
    training.supervised(
        model,
        sup_trainloader,
        sup_testloader,
        config['training']['supervised']['epochs'],
        config['training']['supervised']['lr'],
        device,
        verbose=False
    )

    # evaluate model and log accuracy
    test_acc = training.eval(model, sup_testloader, device)
    msg = f'{activation}-{beta}_beta-{epochs}_epochs-{lr}_lr: {test_acc:.1f}'
    logging.info(msg)