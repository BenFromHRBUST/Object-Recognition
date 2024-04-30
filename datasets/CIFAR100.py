from torchvision import datasets

from .utils import dataset_transform


class CIFAR100:
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

        self.is_downloaded = False     # False when it is first time to run the code, then it will be True
        if not self.is_downloaded:
            self.train_dataset, self.test_dataset = self._download()

    def get_dataset(self):
        return self.train_dataset, self.test_dataset

    def _download(self):
        print("[+] Downloading and transforming CIFAR-100 dataset...")
        train_dataset = datasets.CIFAR100(root=self.dataset_config['general']['root'],
                                         train=True,
                                         download=True,
                                         transform=dataset_transform(config_general=self.dataset_config['general'],
                                                                     config_augmentation=
                                                                     self.dataset_config['datasets']['train'][
                                                                         'augmentation']))
        test_dataset = datasets.CIFAR100(root=self.dataset_config['general']['root'],
                                        train=False,
                                        download=True,
                                        transform=dataset_transform(config_general=self.dataset_config['general'],
                                                                    config_augmentation=
                                                                    self.dataset_config['datasets']['test'][
                                                                        'augmentation']))
        self.is_downloaded = True
        print("[+] Downloading and transforming CIFAR-100 dataset...DONE!")

        return train_dataset, test_dataset
