import os
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
import h5py
import matplotlib.pyplot as plt
import torch

def load_data_from_h5(path='./datasets/cats'):
    # 按照数据组织方式，从*.h5文件中读取字典数据
    train_dataset = h5py.File(os.path.join(path, 'train_catvnoncat.h5'), "r")
    # 训练集数据
    train_set_x = train_dataset['train_set_x'][:].reshape(-1, 64 * 64 * 3).astype(np.float32) / 255.0
    # 训练集标签
    train_set_y = train_dataset['train_set_y'][:]
    train_data = list(zip(train_set_x, train_set_y))

    test_dataset = h5py.File(os.path.join(path, 'test_catvnoncat.h5'), "r")
    # 测试集数据
    test_set_x = test_dataset['test_set_x'][:].reshape(-1, 64 * 64 * 3).astype(np.float32) / 255.0
    # 测试集标签
    test_set_y = test_dataset['test_set_y'][:]
    test_data = list(zip(test_set_x, test_set_y))
    # 类型名列表
    classes = np.array(test_dataset["list_classes"][:])
    # 将训练数据和测试数据的标注编成行向量
    print(f'Shape of Train Set: x--{train_set_x.shape}  y--{train_set_y.shape}')
    print(f'Shape of Test Set: x--{test_set_x.shape}  y--{test_set_y.shape}')
    print(classes)
    return train_data, test_data, classes


def load_data_from_folder(path='./datasets/cats-images'):
    transform = transforms.Compose([transforms.ToTensor()]) #,
                        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    train_data = ImageFolder(os.path.join(path, 'train'), transform=transform)
    test_data = ImageFolder(os.path.join(path, 'test'), transform=transform)
    return train_data, test_data, train_data.classes

def save_images(path='./datasets/cats'):
    train_dataset = h5py.File(os.path.join(path, 'train_catvnoncat.h5'), "r")
    train_set_x = train_dataset['train_set_x'][:]
    train_set_y = train_dataset['train_set_y'][:]
    for i in range(train_set_y.shape[0]):
        if train_set_y[i] == 0:
            plt.imsave('./datasets/cats-images/train/0/'+str(i)+'.png', train_set_x[i])
        else:
            plt.imsave('./datasets/cats-images/train/1/'+str(i)+'.png', train_set_x[i])
    train_dataset.close()

    test_dataset = h5py.File(os.path.join(path, 'test_catvnoncat.h5'), "r")
    test_set_x = test_dataset['test_set_x'][:]
    test_set_y = test_dataset['test_set_y'][:]
    for i in range(test_set_y.shape[0]):
        if test_set_y[i] == 0:
            plt.imsave('./datasets/cats-images/test/0/'+str(i)+'.png', test_set_x[i])
        else:
            plt.imsave('./datasets/cats-images/test/1/'+str(i)+'.png', test_set_x[i])

    # 类型名列表
    classes = np.array(test_dataset["list_classes"][:])
    test_dataset.close()
    with open('./datasets/cats-images/classes.txt', 'w') as f:
        f.write(str(list(classes)))


import cv2
def data_augument(path='./datasets/cats-images/train/1'):
    base = 300
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        n = np.random.randint(0,2)
        img = cv2.flip(img, n)
        cv2.imwrite(os.path.join(path,str(base)+'.png'), img)
        base += 1


def draw_metrics(train_loss, test_loss, test_acc):
    # 将 PyTorch tensor 转换为 numpy 数组
    if torch.is_tensor(train_loss):
        train_loss = train_loss.detach().numpy()
    if torch.is_tensor(test_loss):
        test_loss = test_loss.detach().numpy()
    if torch.is_tensor(test_acc):
        test_acc = test_acc.detach().numpy()
    
    plt.figure(figsize=(12, 4))
    plt.subplot(121, title='Loss')
    plt.plot(range(len(train_loss)), train_loss, color='blue', label='Train Loss')
    plt.plot(range(len(train_loss)), test_loss, color='red', label='Test Loss')
    plt.legend()
    plt.subplot(122, title='Test Accuracy')
    plt.plot(range(len(test_acc)), test_acc, color='r')
    plt.show()


if __name__ == '__main__':
    data_augument()
