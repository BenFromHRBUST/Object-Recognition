config_wandb = {
    'api_key': '',  # Add your API key here
    'project': 'test',
}

config_cifar100 = {
    'train': {
        'batch_size': 128,
        'shuffle': True,
        'num_workers': 8,
    },
    'val': {
        'batch_size': 128,
        'shuffle': False,
        'num_workers': 8,
    },
    'test': {
        'batch_size': 128,
        'shuffle': False,
        'num_workers': 8,
    },
}