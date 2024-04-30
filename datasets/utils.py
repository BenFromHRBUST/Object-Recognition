from torchvision import transforms


def dataset_transform(config_general={}, config_augmentation={}):
    transform_list = []

    if 'resize' in config_general:
        transform_list.append(transforms.Resize(config_general['resize']))

    if ('flip' in config_augmentation) and config_augmentation['flip']:
        transform_list.append(transforms.RandomHorizontalFlip())

    if ('crop' in config_augmentation) and config_augmentation['crop']:
        transform_list.append(transforms.RandomCrop(32, padding=4))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)
