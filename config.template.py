WANDB_API_KEY = ''    # Your API key
WANDB_ENTITY = 'bugmakerh'

default_program_config = {
    'production': True,  # Set this to 'True' if you want to run the code in production mode
    'mode': 'sweep',    # 'train' or 'sweep'
    'model': 'ImprovedCNN',  # 'SimpleCNN' or 'ImprovedCNN'
    'dataset': 'CIFAR100',  # 'CIFAR100'
    'wandb': {
        'api_key': WANDB_API_KEY,
        'project': 'AAA',
        'entity': WANDB_ENTITY,
    },
}

default_train_config = {
    'general': {
        'epochs': 200,
        'optimizer': 'Adam',
        'batch_size': 256,
        'learning_rate': 0.001,
        'weight_decay': 0.001,
        'fig_path': './fig',
    },
    'SimpleCNN': {
        'activation_function': 'LeakyReLU',
    },
    'ImprovedCNN': {
        'dropout_rate': 0.5,
        'activation_function': 'LeakyReLU',
    },
}

default_dataset_config = {
    'CIFAR100': {
        'root': './tmp_dataset',
        'datasets': {
            'train': {
                'shuffle': True,
                'num_workers': 8,
                'augmentation': {
                    'flip': True,
                    'crop': True,
                },
            },
            'val': {
                'ratio': 0.2,  # 20% of the training dataset will be used for validation
                'shuffle': False,
                'num_workers': 8,
            },
            'test': {
                'shuffle': False,
                'num_workers': 8,
            },
        },
    },
}

default_sweep_config = {
    'count': 3,
    'config': {
        'SimpleCNN': {
            'method': 'bayes',
            'metric': {
                'name': 'val_accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'learning_rate': {
                    'min': 1e-4,
                    'max': 1e-2
                },
                'batch_size': {
                    'values': [64, 128, 256]
                },
                'epochs': {
                    'value': 200
                },
                'optimizer': {
                    'values': ['Adam', 'SGD']
                },
                'activation_function': {
                    'values': ['ReLU', 'LeakyReLU']
                },
            }
        },
        'ImprovedCNN': {
            'method': 'bayes',
            'metric': {
                'name': 'val_accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'learning_rate': {
                    'min': 1e-4,
                    'max': 1e-2
                },
                'batch_size': {
                    'values': [64, 128, 256]
                },
                'epochs': {
                    'value': 3
                },
                'optimizer': {
                    'values': ['Adam', 'SGD']
                },
                'activation_function': {
                    'values': ['ReLU', 'LeakyReLU']
                },
                'dropout_rate': {
                    'min': 0.0,
                    'max': 0.5
                }
            }
        }
    }
}
