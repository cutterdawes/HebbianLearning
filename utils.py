import yaml
import logging
import os

import torch
from torch.utils.data import DataLoader

from models.genhebb import GenHebb
from models.baseline import Baseline
from dataset import FastMNIST


def setup_logging(log_file='train.log'):
    # create a logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(os.path.join('logs', log_file)),
            logging.StreamHandler()  # Logs to console as well
        ]
    )


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(config):
    if config['model']['type'] == 'GenHebb':
        return GenHebb(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            output_dim=config['model']['output_dim'],
            learning_rule=config['model']['learning_rule'],
            n_hebbian_layers=config['model']['n_hebbian_layers'],
            **config['model']['kwargs']
        )
    elif config['model']['type'] == 'Baseline':
        return Baseline(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            output_dim=config['model']['output_dim'],
            n_hidden_layers=config['model']['n_hidden_layers'],
            dropout_p=config['model']['dropout_p']
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
    

def load_data(config, train_type, device):
    # load train and test data
    trainset = FastMNIST('./data', train=True, download=True, device=device)
    testset = FastMNIST('./data', train=False, download=True, device=device)

    trainloader = DataLoader(trainset, batch_size=config[train_type]['batch_size'], shuffle=True)
    testloader = DataLoader(testset, batch_size=config[train_type]['batch_size'], shuffle=False)

    if train_type == 'unsup_training':
        return trainloader
    elif train_type == 'sup_training':
        return trainloader, testloader
    else:
        raise KeyError('Invalid train_type')