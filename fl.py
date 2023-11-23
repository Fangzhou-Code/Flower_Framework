'''
MNIST 切分成100份分给100个客户端
'''
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import flwr as fl
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar


# 下载数据集
def get_mnist(data_path: str='./data'):
    '''This function download MNIST dataset into 'data_path' if it is not there already. We construct the train/test split by converting the images into tensors and normalising them'''

    # transformation to convert images to tensors and apply normalisation
    tr = Compose([ToTensor(), Normalize((0.1307),(0.3081,))])

    #prapre the train and test set
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset,testset

# 切分数据集
def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    """This function partitions the training set into N disjoint
    subsets, each will become the local dataset of a client. This
    function also subsequently partitions each traininset partition
    into train and validation. The test set is left intact and will
    be used by the central server to asses the performance of the
    global model."""

    # get the MNIST dataset
    trainset, testset = get_mnist()

    # split trainset into `num_partitions` trainsets
    num_images = len(trainset) // num_partitions

    partition_len = [num_images] * num_partitions

    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    # create dataloader for the test set
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader




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

# 模型测试
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

# 定义Flower客户端
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, vallodaer) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = vallodaer
        self.model = Net(num_classes=10)

    def set_parameters(self, parameters):
        """With the model parameters received from the server,
        overwrite the uninitialise model in this class with them."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        # now replace the parameters
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract all model parameters and convert them to a list of
        NumPy arrays. The server doesn't work with PyTorch/TF/etc."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """This method train the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # Define the optimizer -------------------------------------------------------------- Essentially the same as in the centralised example above
        optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # do local training  -------------------------------------------------------------- Essentially the same as in the centralised example above (but now using the client's data instead of the whole dataset)
        train(self.model, self.trainloader, optim, epochs=1)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        self.set_parameters(parameters)
        loss, accuracy = test(
            self.model, self.valloader
        )  # <-------------------------- calls the `test` function, just what we did in the centralised setting (but this time using the client's local validation set)
        # send statistics back to the server
        return float(loss), len(self.valloader), {"accuracy": accuracy}


# 选择Flower策略
def get_evalulate_fn(testloader):
    """This is a function that returns a function. The returned
    function (i.e. `evaluate_fn`) will be executed by the strategy
    at the end of each round to evaluate the stat of the global
    model."""

    def evaluate_fn(server_round: int, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = Net(num_classes=10)

        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # call test
        loss, accuracy = test(
            model, testloader
        )  # <-------------------------- calls the `test` function, just what we did in the centralised setting
        return loss, {"accuracy": accuracy}

    return evaluate_fn




# 创建一个回调，该回调由Simulation Engine 使用以生成虚拟客户端(VirtualClient)
def generate_client_fn(trainloaders, valloaders):
    def client_fn(cid: str):
        """Returns a FlowerClient containing the cid-th data partition"""

        return FlowerClient(
            trainloader=trainloaders[int(cid)], vallodaer=valloaders[int(cid)]
        )

    return client_fn





if __name__=='__main__':
    trainloaders, valloaders, testloader = prepare_dataset(num_partitions=100, batch_size=32)

    '''
    抓取第一个分区看下。
    这个直方图和整体数据的直方图区别不大，因为是独立同分布采样。
    现实中可能是non-IID分布
    '''
    # # first partition
    # train_partition = trainloaders[0].dataset
    #
    # # count data points
    # partition_indices = train_partition.indices
    # print(f"number of images: {len(partition_indices)}")
    #
    # # visualise histogram
    # # plt.hist(train_partition.dataset.dataset.targets[partition_indices], bins=10)
    # # plt.grid()
    # # plt.xticks(range(10))
    # # plt.xlabel ('Label')
    # # plt.ylabel('Number of images')
    # # plt.title('Class labels distribution for MNIST')
    # # plt.show()

    # now we can define the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # let's sample 10% of the client each round to do local training
        fraction_evaluate=0.1,
        # after each round, let's sample 20% of the clients to asses how well the global model is doing
        min_available_clients=100,  # total number of clients available in the experiment
        evaluate_fn=get_evalulate_fn(testloader),
    )  # a callback to a function that the strategy can execute to evaluate the state of the global model on a centralised dataset

    # 创建一个flowerclient对象，并为每个对象分配自己的数据分区
    client_fn_callback = generate_client_fn(trainloaders, valloaders)

    # 使用flower的仿真来启动FL
    history = fl.simulation.start_simulation(
        client_fn=client_fn_callback,  # a callback to construct a client
        num_clients=100,  # total number of clients in the experiment
        config=fl.server.ServerConfig(num_rounds=10),  # let's run for 10 rounds
        strategy=strategy,  # the strategy that will orchestrate the whole FL pipeline
    )

    # 可视化
    print(f"{history.metrics_centralized = }")

    global_accuracy_centralised = history.metrics_centralized["accuracy"]
    round = [data[0] for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]
    plt.plot(round, acc)
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title("MNIST - IID - 100 clients with 10 clients per round")
    plt.show()