# This is a template file for the configuration file.
# You can copy this file and rename it to 'config.py' to use it.
# You can modify the default configurations as you want.

# Wandb sensitive information. Please do not share this file with others to protect your privacy.
WANDB_API_KEY = ''    # Your API key. You can get it from https://wandb.ai/authorize
WANDB_ENTITY = 'bugmakerh'  # Your entity name. You can find it in the URL of your wandb project page.

# Default program configurations
default_program_config = {
    'production': True,  # [True, False]. Set this to 'True' if you want to run the code in production mode. But it must be 'True' when you run the code in sweep mode.
    'mode': 'sweep',    # ['train', 'sweep']. 'train' is for training a single model, and 'sweep' is for hyperparameter tuning.
    'model': 'SimpleCNN',  # ['SimpleCNN', 'EnhancedCNN', 'ImprovedCNN', 'InceptionLikeCNN', 'ResidualCNN', 'SimpleAlexNet'].
    'dataset': 'CIFAR10',  # ['CIFAR10', 'CIFAR100']
    'wandb': {
        'api_key': WANDB_API_KEY,
        'project': 'Test',   # Your project name in wandb
        'entity': WANDB_ENTITY,
    },
}

# Default training configurations.
# - You just need to modify the model you want to train and ignore the other models.
# - If you are going to use the sweep mode,
#       you need not modify items who are already setting in the sweep configuration.
#       Because the sweep configuration will override the default training configuration.
default_train_config = {
    # General configurations
    'general': {
        'epochs': 200,  # Number of epochs
        'optimizer': 'SGD',    # Optimizer to use. For more optimizers, you can check the 'torch.optim' module from https://pytorch.org/docs/stable/optim.html#algorithms.
        'batch_size': 256,  # Batch size
        'learning_rate': 0.001,   # Learning rate
        'weight_decay': 0.001,  # Weight decay
        'fig_path': './fig',    # Path to save the figures
    },
    # Model-specific configurations
    'SimpleCNN': {
        'activation_function': 'LeakyReLU', # Activation function to use. For more activation functions, you can check the 'torch.nn' module from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
    },
    'EnhancedCNN': {
        'dropout_rate': 0.5,    # Dropout rate
        'activation_function': 'LeakyReLU', # Activation function to use. For more activation functions, you can check the 'torch.nn' module from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
    },
    'ImprovedCNN': {
        'dropout_rate': 0.5,    # Dropout rate
        'activation_function': 'LeakyReLU', # Activation function to use. For more activation functions, you can check the 'torch.nn' module from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
    },
    'InceptionLikeCNN': {
        'activation_function': 'LeakyReLU', # Activation function to use. For more activation functions, you can check the 'torch.nn' module from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
    },
    'ResidualCNN': {
        'activation_function': 'LeakyReLU', # Activation function to use. For more activation functions, you can check the 'torch.nn' module from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
    },
    'SimpleAlexNet': {
        'activation_function': 'LeakyReLU',
    }
}

# Default dataset configurations
default_dataset_config = {
    'CIFAR10': {
        'general': {
            'root': './tmp_dataset',    # Path to save the dataset
            'resize': (224, 224),   # Resize the image to fit the model
            'num_workers': 8,   # Number of workers
        },
        'datasets': {
            # Training dataset configurations
            'train': {
                'shuffle': True,    # Shuffle the dataset
                'num_workers': 8,   # Number of workers
                # Data augmentation
                'augmentation': {
                    'flip': False,   # Randomly flip the image horizontally
                    'crop': False,   # Randomly crop the image
                },
            },
            # Validation dataset configurations
            'val': {
                'ratio': 0.2,  # Validation ratio. 0.2 means 20% of the training dataset will be used for validation.
                'shuffle': False,   # Shuffle the dataset
            },
            # Test dataset configurations
            'test': {
                'shuffle': False,   # Shuffle the dataset
                'num_workers': 8,   # Number of workers
                'augmentation': {
                },
            },
        },
    },
    'CIFAR100': {
        'general': {
            'root': './tmp_dataset',    # Path to save the dataset
            'num_workers': 8,   # Number of workers
        },
        'datasets': {
            # Training dataset configurations
            'train': {
                'shuffle': True,    # Shuffle the dataset
                'num_workers': 8,   # Number of workers
                # Data augmentation
                'augmentation': {
                    'flip': True,   # Randomly flip the image horizontally
                    'crop': True,   # Randomly crop the image
                },
            },
            # Validation dataset configurations
            'val': {
                'ratio': 0.2,  # Validation ratio. 0.2 means 20% of the training dataset will be used for validation.
                'shuffle': False,   # Shuffle the dataset
            },
            # Test dataset configurations
            'test': {
                'shuffle': False,   # Shuffle the dataset
                'num_workers': 8,   # Number of workers
            },
        },
    },
}

# Default sweep configurations
default_sweep_config = {
    'count': 50,     # Number of runs
    'config': {
        # Model-specific configurations
        'SimpleCNN': {
            'method': 'bayes',  # ['grid', 'random', 'bayes']. Method to use for hyperparameter tuning.
            'metric': {
                'name': 'val_accuracy',   # Metric to use for hyperparameter tuning
                'goal': 'maximize'  # ['maximize', 'minimize']. Goal of the metric
            },
            # Hyperparameters to tune
            'parameters': {
                'learning_rate': {
                    'min': 1e-4,    # Minimum value
                    'max': 1e-2    # Maximum value
                },
                'batch_size': {
                    'values': [64, 128, 256]
                },
                'epochs': {
                    'value': 100
                },
                'optimizer': {
                    'values': ['Adam', 'SGD']   # For more optimizers, you can check the 'torch.optim' module from https://pytorch.org/docs/stable/optim.html#algorithms.
                },
                'activation_function': {
                    'values': ['ReLU', 'LeakyReLU']   # For more activation functions, you can check the 'torch.nn' module from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
                },
            }
        },
        'EnhancedCNN': {
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
                    'value': 100
                },
                'optimizer': {
                    'values': ['Adam', 'SGD']   # For more optimizers, you can check the 'torch.optim' module from https://pytorch.org/docs/stable/optim.html#algorithms.
                },
                'activation_function': {
                    'values': ['ReLU', 'LeakyReLU']   # For more activation functions, you can check the 'torch.nn' module from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
                },
                'dropout_rate': {
                    'min': 0.0,
                    'max': 0.5
                }
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
                    'value': 100
                },
                'optimizer': {
                    'values': ['Adam', 'SGD']   # For more optimizers, you can check the 'torch.optim' module from https://pytorch.org/docs/stable/optim.html#algorithms.
                },
                'activation_function': {
                    'values': ['ReLU', 'LeakyReLU']  # For more activation functions, you can check the 'torch.nn' module from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
                },
                'dropout_rate': {
                    'min': 0.0,
                    'max': 0.5
                }
            }
        },
        'InceptionLikeCNN': {
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
                    'value': 100
                },
                'optimizer': {
                    'values': ['Adam', 'SGD']   # For more optimizers, you can check the 'torch.optim' module from https://pytorch.org/docs/stable/optim.html#algorithms.
                },
                'activation_function': {
                    'values': ['ReLU', 'LeakyReLU']   # For more activation functions, you can check the 'torch.nn' module from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
                },
            }
        },
        'ResidualCNN': {
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
                    'value': 100
                },
                'optimizer': {
                    'values': ['Adam', 'SGD']   # For more optimizers, you can check the 'torch.optim' module from https://pytorch.org/docs/stable/optim.html#algorithms.
                },
                'activation_function': {
                    'values': ['ReLU', 'LeakyReLU']   # For more activation functions, you can check the 'torch.nn' module from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
                },
            }
        },
        'SimpleAlexNet': {
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
                    'value': 100
                },
                'optimizer': {
                    'values': ['Adam', 'SGD']   # For more optimizers, you can check the 'torch.optim' module from https://pytorch.org/docs/stable/optim.html#algorithms.
                },
                'activation_function': {
                    'values': ['ReLU', 'LeakyReLU']   # For more activation functions, you can check the 'torch.nn' module from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
                },
            }
        },
    }
}
