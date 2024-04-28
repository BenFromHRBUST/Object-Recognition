import copy

import wandb

from train import train
from config import (default_program_config,
                    default_train_config,
                    default_dataset_config,
                    default_sweep_config)


def run_sweep(program_config, train_config, dataset_config):
    is_production = True

    wandb.init() if is_production else None

    model_name = program_config['model']
    for key, value in wandb.config.items():
        print("DEBUG: ", type(train_config['general']))
        if key in train_config['general'].keys():
            train_config['general'][key] = value
        elif key in train_config[model_name]:
            train_config[model_name][key] = value

    train(program_config['model'],
          train_config['general'],
          train_config[program_config['model']],
          program_config['dataset'],
          dataset_config[program_config['dataset']],
          is_production=program_config['production'])

    wandb.finish() if is_production else None


def main():
    program_config = default_program_config
    train_config = default_train_config
    dataset_config = default_dataset_config

    if program_config['wandb']['api_key'] == '' and program_config['mode'] == 'sweep':
        raise ValueError("API key is required to run the code in sweep mode.")
    elif program_config['wandb']['api_key'] == '' and program_config['production'] is True:
        print("[!] API key is not provided. Running the code in develop mode...")
        program_config['production'] = False
    elif program_config['mode'] == 'sweep' and program_config['production'] is False:
        print("[!] Production mode must be set to 'True' to run the code in sweep mode. Running the code in production mode...")
        program_config['production'] = True

    wandb.login(key=program_config['wandb']['api_key']) if program_config['production'] else None

    if program_config['mode'] == 'train':
        wandb.init(program_config['wandb']['project']) if program_config['production'] else None
        train(program_config['model'],
              train_config['general'],
              train_config[program_config['model']],
              program_config['dataset'],
              dataset_config[program_config['dataset']],
              is_production=program_config['production'])
        wandb.finish() if program_config['production'] else None
    elif program_config['mode'] == 'sweep':
        sweep_config = default_sweep_config
        sweep_id = wandb.sweep(sweep_config['config'][program_config['model']],
                               project=program_config['wandb']['project'])
        wandb.agent(sweep_id,
                    function=lambda: run_sweep(program_config,
                                               train_config,
                                               dataset_config),
                    count=sweep_config['count'])


if __name__ == '__main__':
    print("[+] main.py is running...")
    main()
    print("[+] main.py is done!")
