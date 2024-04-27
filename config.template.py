import models

config = {
    'wandb': {
        'api_key': '',  # Your API key
        'project': 'test',
    },
    'train': {
        'model': models.ImprovedCNN,
        'optimizer': 'Adam',
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 0.001,

        'fig_path': './fig',
    },
    'cifar100': {
        'root': './tmp_dataset',
        'datasets': {
            'train': {
                'batch_size': 128,
                'shuffle': True,
                'num_workers': 8,
                'augmentation': {
                    'flip': True,
                    'crop': True,
                },
            },
            'val': {
                'ratio': 0.2,  # 20% of the training dataset will be used for validation
                'batch_size': 128,
                'shuffle': False,
                'num_workers': 8,
            },
            'test': {
                'batch_size': 128,
                'shuffle': False,
                'num_workers': 8,
            },
        },
    },
}
