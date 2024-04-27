import wandb

from utils import check_device
from config import (config_wandb,
                    config_cifar100)


def main():
    wandb.login(key=config_wandb['api_key'])

    device = check_device()


if __name__ == '__main__':
    main()
