API_KEY = ''    # Your API key

default_program_config = {
    'production': False,  # Set this to 'True' if you want to run the code in production mode
    'mode': 'train',    # 'train' or 'sweep'
    'model': 'SimpleCNN',  # 'SimpleCNN'
    'dataset': 'CIFAR100',  # 'CIFAR100'
    'wandb': {
        'api_key': API_KEY,
        'project': 'test',
    },
}


default_train_config = {
    'SimpleCNN': {
        'optimizer': 'Adam',
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
        'dropout_rate': 0.5,
        'weight_decay': 0.001,
        'fig_path': './fig',
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
    'count': 100,
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
                    'value': 10
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
