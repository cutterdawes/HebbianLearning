import argparse
import logging

import torch

import training
from utils import load_config, setup_logging, get_device, get_model, load_data


# Main training loop MNIST
if __name__ == "__main__":

    # create and parse arguments
    parser = argparse.ArgumentParser(description='Train a fully-connected model on MNIST using specified Hebbian learning rule or BP (baseline)')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file (default: config.yml)')
    parser.add_argument('--save', action='store_true', help='Save model')
    args = parser.parse_args()

    # load config
    config = load_config(args.config)

    # setup logging
    setup_logging()

    # get device
    device = get_device()

    # load model
    model = get_model(config)

    # unsupervised training (if using a Hebbian learning rule)
    if config['model']['type'] == 'GenHebb':
        unsup_trainloader = load_data(config, 'unsup_training', device)
        training.unsupervised(model, unsup_trainloader, config['unsup_training']['epochs'], config['unsup_training']['lr'], device)

    # supervised training
    sup_trainloader, sup_testloader = load_data(config, 'sup_training', device)
    training.supervised(model, sup_trainloader, sup_testloader, config['sup_training']['epochs'], config['sup_training']['lr'], device)

    # save model if specified
    if args.save:
        model_name = 'temp'
        path = f'models/saved/done-training/{model_name}.pt'
        torch.save(model.state_dict(), path)
        print(f'Model saved to: {path}')