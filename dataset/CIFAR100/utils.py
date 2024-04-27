from torchvision import transforms


def dataset_transform(config):
    transform_list = []
    if config['flip']:
        transform_list.append(transforms.RandomHorizontalFlip())
    if config['crop']:
        transform_list.append(transforms.RandomCrop(32, padding=4))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)
