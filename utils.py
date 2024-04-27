import torch
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
        print()


def check_cpu_info():
    print(f'[!] CPU Cores: {torch.get_num_threads()}')
    print(f'[!] CPU RAM: {virtual_memory().total / 1e9} GB')
    print()
