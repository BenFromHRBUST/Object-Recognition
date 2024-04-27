import wandb

from dataset.CIFAR100 import CIFAR100

from train import train
from utils import check_device
from config import (config_wandb,
                    config_train,
                    config_cifar100)


IS_PRODUCTION = True


def main():
    wandb.login(key=config_wandb['api_key']) if IS_PRODUCTION else None

    dataset = CIFAR100(config_cifar100)
    train_loader, val_loader, test_loader = dataset.get_loader()

    train(config_train, train_loader, val_loader, test_loader, is_production=IS_PRODUCTION)


if __name__ == '__main__':
    print("[!] main.py is running...")
    main()
    print("[!] main.py is done!")
