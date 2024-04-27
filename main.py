import wandb

from utils import check_device
from config import (config_wandb,
                    config_cifar100)


IS_PRODUCTION = False


def main():
    wandb.login(key=config_wandb['api_key']) if not IS_PRODUCTION else None

    device = check_device()


if __name__ == '__main__':
    main()
