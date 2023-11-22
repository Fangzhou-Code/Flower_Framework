import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_mnist(data_path: str='./data'):
    '''This function download MNIST dataset into 'data_path' if it is not there already. We construct the train/test split by converting the images into tensors and normalising them'''

    # transformation to convert images to tensors and apply normalisation
    tr = Compose([ToTensor(), Normalize((0.1307),(0.3081,))])

    #prapre the train and test set
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset,testset

# # 获取训练集和测试集
# trainset,testset = get_mnist()

# # 直方图查看标签分布
# all_labels = trainset.targets
# num_possible_labels = len(set(all_labels.numpy().tolist()))
# plt.hist(all_labels,bins=num_possible_labels)
#
# plt.xticks(range(num_possible_labels))
# plt.grid()
# plt.xlabel('Label')
# plt.ylabel('Number of images')
# plt.title('Class labels distribution for MNIST')
# plt.show()


# 数据集中可视化32张图片
def visulaise_n_random_examples(trainset_, n: int, verbose:bool = True): # verbose:bool = True 布尔标志，表示是否打印正在显示的索引信息。
    # take n example at random
    idx = list(range(len(trainset_.data))) # 生成索引表
    random.shuffle(idx) # 打乱索引
    idx= idx[:n] # 选择前n个索引
    if verbose: # 打印
        print(f"will display images with idx:{idx}")

    # construct canvas
    num_clos = 8 # 定义图片为8列
    num_rows = int(np.ceil(len(idx)/num_clos)) # 行数=索引总数/列数 np.ceil返回输入值上限
    fig, axs = plt.subplots(figsize=(16, num_rows*2), nrows=num_rows, ncols=num_clos)

    #display images on canvas
    for c_i, i in enumerate(idx): # 遍历idx，提供了索引的顺序号 c_i 和对应的索引值 i。
        # 对于每个图像索引 i，代码通过 axs.flat[c_i] 获取了一个子图，这个子图是图形界面中的一个小区域。
        # .imshow(trainset_.data[i], cmap='gray') 用来在当前的子图上显示数据集中索引为 i 的图像。cmap='gray' 表示将图像以灰度的方式显示。
        axs.flat[c_i].imshow(trainset_.data[i], cmap='gray') # 这段代码的作用是将数据集中随机选取的图像以灰度形式展示在一个图形界面中，每个图像都显示在独立的小区域中。
        print(f"c_i: {c_i},i:{i}")
    plt.show()
# visulaise_n_random_examples(trainset,n=32)


# 搭建CNN模型
class Net(nn.Module):
    def __init__(self, num_classes: int) -> None: # ->None是对函数返回值的注释，这些注释信息都是函数的元信息，保存在function.__annotations__字典中
        super(Net, self).__init__()
        # 定义卷积层，输入通道数为1，输出通道数为6，卷积核大小为5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 定义最大池化层，池化窗口大小为2x2，步幅为2
        self.pool = nn.MaxPool2d(2, 2)
        # 定义第二个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义全连接层，输入特征数为16*4*4，输出特征数为120
        self.fc1 = nn.Linear(16*4*4, 120)
        # 定义第二个全连接层，输入特征数为120，输出特征数为84
        self.fc2 = nn.Linear(120, 84)
        # 定义输出层，输入特征数为84，输出特征数为num_classes
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一个卷积层，经过ReLU激活函数和最大池化层
        x = self.pool(F.relu(self.conv1(x)))
        # 第二个卷积层，经过ReLU激活函数和最大池化层
        x = self.pool(F.relu(self.conv2(x)))
        # 将特征展平，准备输入全连接层
        x = x.view(-1, 16 * 4 * 4)
        # 第一个全连接层，经过ReLU激活函数
        x = F.relu(self.fc1(x))
        # 第二个全连接层，经过ReLU激活函数
        x = F.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x

# 查看模型
# model = Net(num_classes=10)
# num_parameters = sum(value.numel() for value in model.state_dict().values())
# print(f"{num_parameters = }") # 模型参数总量
# print(model.__init__.__annotations__['return']) # 获取函数的返回值，但是python对注释信息和f.__annotations__的一致性，不做检查，不做强制，不做验证！什么都不做。所以当你->返回值与实际返回不一致的时候会打印->返回值


# 模型训练
def train(net, trainloader, optimizer, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    return net

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def run_centralised(epochs: int, lr:float, momentum: float=0.9):
    # instantiate the model
    model = Net(num_classes=10)

    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # get dataset and construct a dataloaders
    trainset, testset = get_mnist()
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128)

    # train for the specified number of epochs
    trained_model = train(model, trainloader, optim, epochs)

    # traning is completed, the evaluate model on the test set
    loss, accuracy = test(trained_model, testloader)
    print(f"{loss=}")
    print(f"{accuracy=}")


if __name__ == '__main__':
    # freeze_support()
    run_centralised(epochs=5, lr=0.01)