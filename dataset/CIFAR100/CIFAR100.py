from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from .utils import dataset_transform


class CIFAR100:
    def __init__(self, config):
        self.config = config

        self.train_dataset, self.test_dataset = self._download()

    def get_loader(self):
        val_size = int(len(self.train_dataset) * self.config['datasets']['val']['ratio'])
        train_size = len(self.train_dataset) - val_size

        train_dataset, val_dataset = random_split(self.train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config['datasets']['train']['batch_size'],
                                  shuffle=self.config['datasets']['train']['shuffle'],
                                  num_workers=self.config['datasets']['train']['num_workers'])
        val_loader = DataLoader(val_dataset,
                                batch_size=self.config['datasets']['val']['batch_size'],
                                shuffle=self.config['datasets']['val']['shuffle'],
                                num_workers=self.config['datasets']['val']['num_workers'])
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.config['datasets']['test']['batch_size'],
                                 shuffle=self.config['datasets']['test']['shuffle'],
                                 num_workers=self.config['datasets']['test']['num_workers'])

        return train_loader, val_loader, test_loader

    def _download(self):
        print("[!] Downloading and transforming CIFAR-100 dataset...")
        train_dataset = datasets.CIFAR100(root=self.config['root'],
                                          train=True,
                                          download=True,
                                          transform=dataset_transform(self.config['datasets']['train']['augmentation']))
        test_dataset = datasets.CIFAR100(root=self.config['root'],
                                         train=False,
                                         download=True,
                                         transform=dataset_transform)
        print("[!] Downloading and transforming CIFAR-100 dataset...DONE!")

        return train_dataset, test_dataset


if __name__ == '__main__':
    CIFAR100({})
