"""
Пример реализации пользовательских функций
"""
import pickle
import random
import copy
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torch.optim as optim
from google.colab import drive
from engine import Engine
 
 
def my_upload():
    """
    функция, загружающая данные о последнем сохранении с диска
    """
    drive.mount('/content/gdrive')
    print("write filename")
    filename = input()
    a_file = open(
        "/content/gdrive/My Drive/backup/ISearch/SGD_new/" + filename, "rb")
    my_files = pickle.load(a_file)
    return my_files
 
 
def my_backup(backup_data, index):
    """
    Функция, сохраняющая данные на диск
    """
    drive.mount('/content/gdrive')
    a_file = open(
        "/content/gdrive/My Drive/backup/ISearch/SGD_new/" +
        "backup " + str(index) + ".pkl", "wb")
    pickle.dump(backup_data, a_file)
 
 
def teach(point, stop_signal, seed=None):
    """
    Обучает сеть с гиперпараметрами передаваемыми в point,
    возвращает статистики обучения.
    Останавливает обучение, по сигналу функции stop_signal
    Можно задать seed.
    """
    random.seed(seed, version=2)
 
    def test_accuracy(net, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
 
        return 100 * correct / total
 
    def train_accuracy(net, trainloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
 
        return 100 * correct / total
 
    def min_max_average_norm(parameters):
        weight_min = []
        weight_max = []
        weight_average = []
        weight_norm = []
        for _, param in parameters:
            if param.requires_grad:
                weight_min.append(param.min().item())
                weight_max.append(param.max().item())
                weight_average.append(torch.mean(param).item())
                weight_norm.append(param.norm().item())
        return weight_min, weight_max, weight_average, weight_norm
 
    def min_max_average_norm_grad(gradient):
        grad_min = []
        grad_max = []
        grad_average = []
        grad_norm = []
        for grad in gradient:
            grad_min.append(grad.min().item())
            grad_max.append(grad.max().item())
            grad_average.append(torch.mean(grad).item())
            grad_norm.append(grad.norm().item())
        return grad_min, grad_max, grad_average, grad_norm
 
    def grad(net):
        gradient = []
        for _, param in net.named_parameters():
            if param.grad is not None:
                gradient.append(param.grad)
        return gradient
 
    def test_loss(net, testloader, criterion):
        running_loss = 0
        batch_num = 0
        for data in testloader:
            batch_num += 1
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        return running_loss/batch_num
 
    class SGDnew(optim.SGD):
        """
        Реализация нормализованного импульса.
        Градиенты домножаются на (1 - \beta), \beta - momentum
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
 
        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
 
            for group in self.param_groups:
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                dampening = group['dampening']
                nesterov = group['nesterov']
 
                for param in group['params']:
                    if param.grad is None:
                        continue
                    d_p = param.grad*(1-momentum)
                    if weight_decay != 0:
                        d_p = d_p.add(param, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[param]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state[
                                'momentum_buffer'
                                ] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
 
                    param.add_(d_p, alpha=-group['lr'])
 
            return loss
 
    def backup(net, i_global, error, running_loss,
               running_loss_id, optimizer,
               criterion, testloader, trainloader, data):
        cur_er = test_accuracy(net, testloader)
        data["x"].append(i_global)
        data["train_accuracy"].append(train_accuracy(net, trainloader))
 
        data["lr_scheduler"].append(optimizer.param_groups[0]['lr'])
        data["momentum_cycle"].append(optimizer.param_groups[0]['momentum'])
        data["lr_momentum"].append(
            data["lr_scheduler"][-1]/(1-data["momentum_cycle"][-1])
        )
        data["test_loss"].append(test_loss(net, testloader, criterion))
        data["test_accuracy"].append(cur_er)
        data["train_loss"].append(running_loss/running_loss_id)
        min_, max_, average_, norm_ = min_max_average_norm(
            net.named_parameters())
        data["weight_min"].append(min_)
        data["weight_max"].append(max_)
        data["weight_average"].append(average_)
        data["weight_norm"].append(norm_)
        min_, max_, average_, norm_ = min_max_average_norm_grad(grad(net))
        data["grad_min"].append(min_)
        data["grad_max"].append(max_)
        data["grad_average"].append(average_)
        data["grad_norm"].append(norm_)
 
        path = './cifar_net.pth'
        if cur_er > error:
            torch.save(net.state_dict(), path)
            error = cur_er
        data["accuracy"] = error
        print(f'Accuracy of the network on the 10000 test images: {cur_er}%')
        print('Loss of the network on the 10000 test images: ',
              f'{data["test_loss"][-1]}')
        return error, data
 
    transform = {
        'train': transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
            )
        ]),
        'test': transforms.Compose([
            transforms.Pad(1),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
            )
        ])
    }
 
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True,
                                            transform=transform['train'])
 
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True,
                                           transform=transform['test'])
    batchsize = 1024  # batch
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                             shuffle=False, num_workers=2)
 
    device = torch.device('cuda:0')
    # 3x3 convolution
    def conv3x3(in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=False)
 
    # Residual block
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels,
                     stride=1, downsample=None):
            super(ResidualBlock, self).__init__()
            self.conv1 = conv3x3(
                in_channels, out_channels, stride)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(
                out_channels, out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample
 
        def forward(self, inputs):
            residual = inputs
            out = self.conv1(inputs)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample:
                residual = self.downsample(inputs)
            out += residual
            out = self.relu(out)
            return out
 
    # ResNet
    class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes=10):
            super(ResNet, self).__init__()
            self.in_channels = 16
            self.conv = conv3x3(3, 16)
            self.batch_norm = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self.make_layer(block, 16, layers[0])
            self.layer2 = self.make_layer(block, 32, layers[1], 2)
            self.layer3 = self.make_layer(block, 64, layers[2], 2)
            self.avg_pool = nn.AvgPool2d(8)
            self.fully_connected = nn.Linear(64, num_classes)
 
        def make_layer(self, block, out_channels, blocks, stride=1):
            downsample = None
            if (stride != 1) or (self.in_channels != out_channels):
                downsample = nn.Sequential(
                    conv3x3(self.in_channels, out_channels, stride=stride),
                    nn.BatchNorm2d(out_channels))
            layers = []
            layers.append(
                block(self.in_channels, out_channels, stride, downsample)
            )
            self.in_channels = out_channels
            for i in range(1, blocks):
                layers.append(block(out_channels, out_channels))
            return nn.Sequential(*layers)
 
        def forward(self, inputs):
            out = self.conv(inputs)
            out = self.batch_norm(out)
            out = self.relu(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fully_connected(out)
            return out
 
    net = ResNet(ResidualBlock, [2, 2, 2]).to(device)
 
    criterion = nn.CrossEntropyLoss()
    learning_rate = point["lr"]
    if point["warm_up"] == 0:
        start = point["lr"]
    else:
        start = point["start_lr"]
    optimizer = SGDnew(net.parameters(),
                       lr=start,
                       momentum=point["momentum"],
                       weight_decay=point["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=point[
                                                               "patience"
                                                           ],
                                                           threshold=point[
                                                               "threshold"
                                                           ],
                                                           factor=point[
                                                               "factor"
                                                           ],
                                                           mode=point[
                                                               "mode"
                                                           ])
 
    statistics = {
        "accuracy": 0,
        "x": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "train_loss": [],
        "test_loss": [],
        "grad_min": [],
        "grad_max": [],
        "grad_average": [],
        "grad_norm": [],
        "weight_min": [],
        "weight_max": [],
        "weight_average": [],
        "weight_norm": [],
        "lr_scheduler": [],
        "momentum_cycle": [],
        "lr_momentum": []
    }
    error = 0
    i_global = 0
    outputs = 0
    inc_epoch = point["warm_up"]
    epoch = 0
    running_loss_id = 0
    running_loss = 0.0
 
    # For updating learning rate
    def update_lr(optimizer, end, epoch, start):
        for param_group in optimizer.param_groups:
            param_group['lr'] += (end-start)/epoch
 
    while True:  # loop over the dataset multiple times
        print(epoch)
        if epoch < inc_epoch:
            update_lr(optimizer, learning_rate, inc_epoch, start)
        elif epoch != 0:
            scheduler.step(statistics[point["parameter"]][-1])
        for data in trainloader:
            running_loss_id += 1
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            i_global += 1
        error, statistics = backup(net, i_global, error,
                                   running_loss, running_loss_id,
                                   optimizer, criterion, testloader,
                                   trainloader, statistics)
        running_loss_id = 0
        running_loss = 0.0
        if stop_signal(statistics, point["flag"], point["patience"] + 5,
                       point["parameter"], point["epochs"]):
            break
        epoch += 1
    print('Finished Training')
 
    return copy.deepcopy({'data': statistics,
                          'model': {'state_dict': net.state_dict(),
                                    'optimizer': optimizer.state_dict()}})
 
 
isearch = Engine(teach, my_upload, my_backup)
isearch.start()
