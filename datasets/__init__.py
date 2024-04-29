from torch.utils.data import DataLoader, random_split

from .CIFAR100 import CIFAR100


class Datasets:
    def __init__(self, dataset_name, dataset_config):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

        dataset = eval(dataset_name)(dataset_config)
        self.train_dataset, self.test_dataset = dataset.get_dataset()

    def get_loader(self, batch_size=64):
        val_size = int(len(self.train_dataset) * self.dataset_config['datasets']['val']['ratio'])
        train_size = len(self.train_dataset) - val_size

        train_dataset, val_dataset = random_split(self.train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=self.dataset_config['datasets']['train']['shuffle'],
                                  num_workers=self.dataset_config['general']['num_workers'])
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=self.dataset_config['datasets']['val']['shuffle'],
                                num_workers=self.dataset_config['general']['num_workers'])
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=batch_size,
                                 shuffle=self.dataset_config['datasets']['test']['shuffle'],
                                 num_workers=self.dataset_config['general']['num_workers'])

        return train_loader, val_loader, test_loader
