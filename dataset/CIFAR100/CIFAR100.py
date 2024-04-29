from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from .utils import dataset_transform


class CIFAR100:
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

        self.is_downloaded = False     # False when it is first time to run the code, then it will be True
        if not self.is_downloaded:
            self.train_dataset, self.test_dataset = self._download()

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

    def _download(self):
        print("[+] Downloading and transforming CIFAR-100 dataset...")
        train_dataset = datasets.CIFAR100(root=self.dataset_config['general']['root'],
                                          train=True,
                                          download=True,
                                          transform=dataset_transform(config=self.dataset_config['datasets']['train']['augmentation']))
        test_dataset = datasets.CIFAR100(root=self.dataset_config['general']['root'],
                                         train=False,
                                         download=True,
                                         transform=dataset_transform())
        self.is_downloaded = True
        print("[+] Downloading and transforming CIFAR-100 dataset...DONE!")

        return train_dataset, test_dataset


if __name__ == '__main__':
    CIFAR100({})
