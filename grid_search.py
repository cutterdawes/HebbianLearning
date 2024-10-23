import logging
import itertools
import yaml

import training
import utils


def run_trial(config):
    # get model and device
    model = utils.get_model(config['model'])
    device = utils.get_device()

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

    return test_acc


# STDP grid search MNIST
if __name__ == '__main__':

    # setup logging
    utils.setup_logging(log_file='grid_search.log')

    # hyperparameters to search
    logging.info('\nSTDP hyperparameter grid...\n')
    param_grid = {
        'activation': ['tanh', 'relu'],
        'layers': [1, 2, 3],
        'importance_factor': [0, 10, 100],
        'beta': [0.1],
        'epochs': [3],
        'lr': [0.0001, 0.0002, 0.0003],
    }
    msg = yaml.dump(param_grid, sort_keys=False, default_flow_style=False)
    logging.info(msg)

    # grid search
    logging.info('\nSearching...\n')
    for activation, layers, importance_factor, beta, epochs, lr in itertools.product(
            param_grid['activation'],
            param_grid['layers'],
            param_grid['importance_factor'],
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
                'n_hebbian_layers': layers,
                'importance_factor': importance_factor,
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

        # run trials
        num_trials = 1
        test_acc = 0
        for _ in range(num_trials):
            test_acc += run_trial(config)
        test_acc /= num_trials

        # log accuracy
        msg = f'{activation}-{layers}_layers-{importance_factor}_importance_factor-{beta}_beta-{epochs}_epochs-{lr}_lr: {test_acc:.1f}'
        logging.info(msg)