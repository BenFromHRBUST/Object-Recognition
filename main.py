import wandb

from dataset.CIFAR100 import CIFAR100

from train import train
from utils import check_device
from config import default_config


def main():
    config = default_config

    if config['wandb']['api_key'] == '':
        config['production'] = False

    wandb.login(key=config['wandb']['api_key']) if config['production'] else None

    dataset = CIFAR100(config['cifar100'])
    train_loader, val_loader, test_loader = dataset.get_loader()

    train(config, train_loader, val_loader, test_loader, is_production=config['production'])


if __name__ == '__main__':
    print("[!] main.py is running...")
    main()
    print("[!] main.py is done!")
