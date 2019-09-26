from __future__ import print_function
import time
import torch
import math
import random
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
###classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
##########################################################################
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic=True



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
test_bs = 128
##########################################################################
mean = [0.49139968, 0.48215827, 0.44653124]
std = [0.24703233, 0.24348505, 0.26158768]



train_tfm = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])

test_tfm = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])

Train_Set = datasets.CIFAR10(root='./data', train=True, transform=train_tfm, download=True)
Test_Set = datasets.CIFAR10(root='./data', train=False, transform=test_tfm, download=True)#生成一个个batch进行批训练，组成batch的时候顺序打乱取

train_loader = DataLoader(Train_Set, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

test_loader = DataLoader(Test_Set, batch_size=test_bs, shuffle=False,
                      num_workers=0, pin_memory=True)
##########################################################################
def train(model, data_loader, optimizer, epoch, verbose=True):
    model.train()
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss   = F.cross_entropy(output, target)
        loss_avg += loss.item()
        loss.backward()
        optimizer.step()
        verbose_step = len(data_loader) // 10
        if batch_idx % verbose_step == 0 and verbose:
            print('Train Epoch: {}  Step [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    return loss_avg / len(data_loader)

def test(model, data_loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()   

        test_loss /= len(data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))   
    return float(correct) / len(data_loader.dataset)

def adjust_learning_rate(base_lr, optimizer, epoch, decay_step=None, epoch_list=None):

    assert decay_step is None or epoch_list is None, "decay_step and epoch_list can only set one of them"

    if epoch_list is not None:
        index = 0
        for i, e in enumerate(epoch_list, 1):
            if epoch >= e:
                index = i
            else:
                break
        lr = base_lr * (0.1 ** index )
        
    elif decay_step is not None:
        lr = base_lr * (0.1 ** (epoch // decay_step))

    else:
        lr = base_lr * (0.1 ** (epoch // 30))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

##########################################################################
def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class BasicBlock(nn.Module):

    expansion = 1
    
    def __init__(self, inplanes, planes ,stride = 1, downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Resnet(nn.Module):
    
    def __init__(self, block, layers, num_classes=10):
        super(Resnet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)   
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)   
        return x


def Resnet54():
    return Resnet(BasicBlock, [8,9,9])

net = Resnet54().to(device)

# Define Loss and optimizer
learning_rate = 0.1
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

# Train the model
num_epochs = 120

StartTime = time.time()
loss, val_acc, lr_curve = [], [], []
for epoch in range(num_epochs):
    lr = adjust_learning_rate(learning_rate, optimizer, epoch, epoch_list=[80, 100])
    train_loss = train(net, train_loader, optimizer, epoch, verbose=True)
    
    loss.append(train_loss)
    valid_acc  = test(net, test_loader)
    val_acc.append(valid_acc)
    lr_curve.append(lr)


EndTime = time.time()
print('Time Usage: ', str(datetime.timedelta(seconds=int(round(EndTime-StartTime)))))



plt.figure()
plt.plot(loss)
plt.title('Train Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
plt.savefig('Train Loss.png')

plt.figure()
plt.plot(val_acc)
plt.title('Valid Acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
plt.savefig('Valid Acc.png')


plt.figure()
plt.plot(lr_curve)
plt.title('Learning Rate')
plt.xlabel('epochs')
plt.ylabel('lr')
plt.yscale('log')
plt.show()
plt.savefig('Learning Rate.png')






























