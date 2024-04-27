import os.path

import torch

import matplotlib.pyplot as plt

import seaborn as sns

from psutil import virtual_memory


def check_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[!] Using {device}')
    print()

    check_cpu_info()
    if device == 'cuda':
        check_gpu_info()

    return device


def check_gpu_info():
    if torch.cuda.is_available():
        print(f'[!] CUDA Version: {torch.version.cuda}')
        print(f'[!] CUDNN Version: {torch.backends.cudnn.version()}')
        print(f'[!] CUDA Device Count: {torch.cuda.device_count()}')
        print(f'[!] CUDA Current Device: {torch.cuda.current_device()}')
        print(f'[!] CUDA Device Name: {torch.cuda.get_device_name(0)}')
        print(f'[!] CUDA Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB')
        print(f'[!] CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1e9} GB')
        print(f'[!] CUDA Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1e9} GB')
        print(f'[!] CUDA Memory Cached: {torch.cuda.memory_reserved() / 1e9} GB')
        print(f'[!] CUDA Max Memory Cached: {torch.cuda.max_memory_reserved() / 1e9} GB')


def check_cpu_info():
    print(f'[!] CPU Cores: {torch.get_num_threads()}')
    print(f'[!] CPU RAM: {virtual_memory().total / 1e9} GB')


def plot_curves(train_losses, train_accuracies, val_losses, val_accuracies, title='Training Curves', path='./fig'):
    has_val = len(val_losses) != 0 and len(val_accuracies) != 0

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss') if has_val else None
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy') if has_val else None
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.suptitle(title)
    plt.savefig(os.path.join(path, '{}.png'.format(title)))
    plt.show()


def draw_confusion_matrix(cm, title='Confusion Matrix', path='./fig'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(os.path.join(path, '{}.png'.format(title)))
    plt.show()
