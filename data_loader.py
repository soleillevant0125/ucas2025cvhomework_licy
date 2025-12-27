import os
import gzip
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

def load_mnist_images(filename):
    """
    解析 IDX3-ubyte.gz 图像文件
    结构: [magic number] [number of images] [rows] [cols] [pixels...]
    """
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # Reshape: (-1, 1, 28, 28) 适配 PyTorch 格式 (N, C, H, W)
    return data.reshape(-1, 1, 28, 28)

def load_mnist_labels(filename):
    """
    解析 IDX1-ubyte.gz 标签文件
    结构: [magic number] [number of items] [labels...]
    """
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def load_local_data(data_path):
    """
    从本地路径加载 .gz 文件
    data_path: 指向包含 4 个 .gz 文件的目录 
    """
    files = {
        'train_img': 'train-images-idx3-ubyte.gz',
        'train_lbl': 'train-labels-idx1-ubyte.gz',
        'test_img': 't10k-images-idx3-ubyte.gz',
        'test_lbl': 't10k-labels-idx1-ubyte.gz'
    }

    # 完整路径
    paths = {k: os.path.join(data_path, v) for k, v in files.items()}

    # 检查文件是否存在
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到文件: {p}\n")

    # 读取并转换数据
    X_train = load_mnist_images(paths['train_img'])
    y_train = load_mnist_labels(paths['train_lbl'])
    X_test = load_mnist_images(paths['test_img'])
    y_test = load_mnist_labels(paths['test_lbl'])

    return X_train, y_train, X_test, y_test


def get_cnn_loaders(batch_size=64, data_root='./data/fashion'):
    """
    加载数据并返回 PyTorch DataLoader
    """
    # 读取数据
    X_train, y_train, X_test, y_test = load_local_data(data_root)

    # 转换为 Tensor 并归一化(C, H, W) [0.0-1.0]
    
    tensor_x_train = torch.Tensor(X_train).float() / 255.0
    tensor_y_train = torch.LongTensor(y_train)

    tensor_x_test = torch.Tensor(X_test).float() / 255.0
    tensor_y_test = torch.LongTensor(y_test)

    # 封装为 Dataset
    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

    # 创建 Loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader

def get_svm_data(data_root='./data/fashion', limit_train=None, limit_test=None):
    """
    加载数据并返回 Flatten 后的 Numpy 数组
    """
    X_train, y_train, X_test, y_test = load_local_data(data_root)

    # Flatten: (N, 1, 28, 28) -> (N, 784)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 归一化 (SVM 对数值范围敏感)
    X_train_flat = X_train_flat.astype(np.float32) / 255.0
    X_test_flat = X_test_flat.astype(np.float32) / 255.0
    
    # 采样
    if limit_train:
        X_train_flat = X_train_flat[:limit_train]
        y_train = y_train[:limit_train]
    if limit_test:
        X_test_flat = X_test_flat[:limit_test]
        y_test = y_test[:limit_test]

    return X_train_flat, y_train, X_test_flat, y_test