import gc
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils import RunningStats


# Run this once to load the train and test data straight into a dataloader class
# that will provide the batches
def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
  
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
    ])

    if test:
        dataset = datasets.CIFAR10(
          root=data_dir, train=False,
          download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


# CIFAR10 dataset 
train_loader, valid_loader = data_loader(data_dir='./files',
                                         batch_size=64)

test_loader = data_loader(data_dir='./files',
                              batch_size=64,
                              test=True)
                              
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_classes = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PlainNet(nn.Module):
    def __init__(self, output_dim, layers):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layer2 = []
        for _ in range(layers[0]):
            layer2 += [
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()]
        self.layer2 = nn.Sequential(*layer2)
        layer3 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()]
        for _ in range(layers[1]):
            layer3 += [
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()]
        self.layer3 = nn.Sequential(*layer3)
        layer4 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()]
        for _ in range(layers[2]):
            layer4 += [
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()]
        self.layer4 = nn.Sequential(*layer4)
        layer5 = [
            nn.Conv2d(256,512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()]
        for _ in range(layers[3]):
            layer5 += [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()]
        self.layer5 = nn.Sequential(*layer5)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Sequential(
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.Linear(1000, output_dim),
            nn.Softmax())
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

def train(model, num_epochs=10):

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        print ('Epoch [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, loss.item()))
                
        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
        
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

    # Statistics
    model.eval()
    with torch.no_grad():
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                if not name in activation:
                    activation[name] = RunningStats()
                    activation[name] += output.detach().cpu()
                else:
                    activation[name] += output.detach().cpu()
                    #torch.cat((activation[name], output.detach()))
            return hook
        
        count = 0
        for i, layer in enumerate(model.modules()):
            if isinstance(layer, torch.nn.modules.BatchNorm2d):
                count += 1
                layer.register_forward_hook(get_activation(f'{i}'))
        print(f'{count} layers tracked')

        for images, _ in valid_loader:
            # run inference
            model(images.to(device))

        print('Statistics of the network: ')
        for k,v in activation.items():
            print(k, v)


        
def save_model (model, name):
    torch.save (model.state_dict(), f'{name}-sd.bin')

assert(device == 'cuda')

model = PlainNet(num_classes, layers=[4, 3, 3, 3]).to(device) # plain-18
print(f'plain-18: {count_parameters(model)}')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)

train(model)
save_model (model, 'plain-18')

model = PlainNet(num_classes, layers=[6, 7, 11, 5]).to(device) # plain-34
print(f'plain-34: {count_parameters(model)}')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)

train(model)
save_model (model, 'plain-34')

# Step 2: ResNet architecture

# source: https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
# explanation: https://erikgaas.medium.com/resnet-torchvision-bottlenecks-and-layers-not-as-they-seem-145620f93096
    
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        base_width = 64
        width = int(out_channels * (base_width / 64.0))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, width, kernel_size = 1, bias = False),
                        nn.BatchNorm2d(width),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(width, width, kernel_size = 3, stride = stride, padding = 1, bias = False),
                        nn.BatchNorm2d(width),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(width, out_channels * self.expansion, kernel_size = 1, bias = False),
                        nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.stride = stride
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device) #resnet-34
#model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device) #resnet-18
print(f'resnet-34: {count_parameters(model)}')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)

train(model)
save_model (model, 'resnet-34')

# Step 3: ResNet with bottlenecks

model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device) #resnet-50

print(f'resnet-50: {count_parameters(model)}')
#print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)

train(model)
save_model (model, 'resnet-50')

# Homework
# Add Kaiming init to resnet.  Does it help?
# Plot the std of layer responses as in Figure 7 of the paper https://arxiv.org/pdf/1512.03385.pdf

