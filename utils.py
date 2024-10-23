import yaml
import logging
import os
import argparse

import torch
from torch.utils.data import DataLoader

from models.genhebb import GenHebb
from models.baseline import Baseline
from dataset import FastMNIST


def setup_logging(log_file='train.log', verbose=True):
    # create a logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # set up handlers (if verbose, log to console)
    handlers = [logging.FileHandler(os.path.join('logs', log_file))]
    if verbose:
        handlers.append(logging.StreamHandler())

    # configure log
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=handlers
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Train a fully-connected model on MNIST using specified Hebbian learning rule or BP (baseline)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--save', action='store_true', help='Save model')
    args = parser.parse_args()
    return args


def load_config(config):
    config_file = f'configs/{config}.yml'
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    logging.info(f'\nparameters...\n\n' + yaml.dump(config, sort_keys=False, default_flow_style=False))
    return config


# def modify_config(config, args):
#     for param in args.params:
#         if 
#     return config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_config):
    if model_config['type'] == 'GenHebb':
        return GenHebb(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            output_dim=model_config['output_dim'],
            learning_rule=model_config['learning_rule'],
            activation=model_config['activation'],
            n_hebbian_layers=model_config['n_hebbian_layers'],
            importance_factor=model_config['importance_factor'],
            **model_config['kwargs']
        )
    elif model_config['type'] == 'Baseline':
        return Baseline(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            output_dim=model_config['output_dim'],
            n_hidden_layers=model_config['n_hidden_layers'],
            dropout_p=model_config['dropout_p']
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    

def load_data(training_config, type, device):
    # load train and test data
    trainset = FastMNIST('./data', train=True, download=True, device=device)
    testset = FastMNIST('./data', train=False, download=True, device=device)

    trainloader = DataLoader(trainset, batch_size=training_config[type]['batch_size'], shuffle=True)
    testloader = DataLoader(testset, batch_size=training_config[type]['batch_size'], shuffle=False)

    if type == 'unsupervised':
        return trainloader
    elif type == 'supervised':
        return trainloader, testloader
    else:
        raise KeyError('Invalid type')
    
    
def save_model(model):
    model_name = 'temp'
    path = f'models/saved/done-training/{model_name}.pt'
    torch.save(model.state_dict(), path)
    logging.info(f'Model saved to: {path}')