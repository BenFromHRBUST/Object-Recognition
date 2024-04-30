import os.path

import torch

import matplotlib.pyplot as plt

import seaborn as sns

from psutil import virtual_memory


def check_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[+] Using {device}')
    print()

    check_cpu_info()
    if device == 'cuda':
        check_gpu_info()

    return device


def check_gpu_info():
    if torch.cuda.is_available():
        print(f'[+] CUDA Version: {torch.version.cuda}')
        print(f'[+] CUDNN Version: {torch.backends.cudnn.version()}')
        print(f'[+] CUDA Device Count: {torch.cuda.device_count()}')
        print(f'[+] CUDA Current Device: {torch.cuda.current_device()}')
        print(f'[+] CUDA Device Name: {torch.cuda.get_device_name(0)}')
        print(f'[+] CUDA Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB')
        print(f'[+] CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1e9} GB')
        print(f'[+] CUDA Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1e9} GB')
        print(f'[+] CUDA Memory Cached: {torch.cuda.memory_reserved() / 1e9} GB')
        print(f'[+] CUDA Max Memory Cached: {torch.cuda.max_memory_reserved() / 1e9} GB')


def check_cpu_info():
    print(f'[+] CPU Cores: {torch.get_num_threads()}')
    print(f'[+] CPU RAM: {virtual_memory().total / 1e9} GB')


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_curves(train_losses, train_accuracies, val_losses, val_accuracies, title='Training Curves', path='./fig'):
    check_path(path)

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
    check_path(path)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(os.path.join(path, '{}.png'.format(title)))
    plt.show()


def optimizer_params_filter(optimizer_name, params):
    valid_params = {
        'Adadelta': ['lr', 'rho', 'eps', 'weight_decay'],
        'Adagrad': ['lr', 'lr_decay', 'weight_decay', 'initial_accumulator_value', 'eps'],
        'Adam': ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad'],
        'AdamW': ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad'],
        'SparseAdam': ['lr', 'betas', 'eps'],
        'Adamax': ['lr', 'betas', 'eps', 'weight_decay'],
        'ASGD': ['lr', 'lambd', 'alpha', 't0', 'weight_decay'],
        'LBFGS': ['lr', 'max_iter', 'max_eval', 'tolerance_grad', 'tolerance_change', 'history_size', 'line_search_fn'],
        'NAdam': ['lr', 'betas', 'eps', 'weight_decay'],
        'RAdam': ['lr', 'betas', 'eps', 'weight_decay'],
        'RMSprop': ['lr', 'alpha', 'eps', 'weight_decay', 'momentum', 'centered'],
        'Rprop': ['lr'],
        'SGD': ['lr', 'momentum', 'dampening', 'weight_decay', 'nesterov'],
    }
    filtered_params = {k: v for k, v in params.items() if k in valid_params[optimizer_name]}
    return filtered_params
