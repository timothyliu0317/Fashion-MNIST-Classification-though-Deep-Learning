import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_csv = pd.read_csv("D:\\Download\\archive\\fashion-mnist_train.csv")
test_csv = pd.read_csv("D:\\Download\\archive\\fashion-mnist_test.csv")


class FashionDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""
    
    def __init__(self, data, transform = None):
        self.fashionMNIST = list(data.values)
        self.transform = transform
        
        label = []
        image = []
        
        for i in self.fashionMNIST:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)





train_set = FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(train_set, batch_size=100)


'''
train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set,batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=100)
                                               

'''





class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn1=nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3,padding=1)
        self.maxpool1=nn.MaxPool2d(kernel_size=2, stride=2)

        self.cnn2=nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3,stride=1,padding=0)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)


        self.fc1 = nn.Linear(in_features=10*6*6, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

        
    def forward(self, x):
        x = self.cnn1(x)
        x=torch.relu(x)
        x=self.maxpool1(x)
        
        x=self.cnn2(x)
        x=torch.relu(x)
        x=self.maxpool2(x)

        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        
        return x




model = CNN()
model.to(device)

error = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)


num_epochs = 10
count = 0
cost=0

loss_list = []

cost_list=[]
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []




for epoch in range(num_epochs):
    cost=0
    total = 0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
    
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        
        outputs = model(train)
        loss = error(outputs, labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
    
        count += 1
        cost+=loss.cpu().detach().numpy()
    
       
        
    for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels_list.append(labels)
            
            test = Variable(images.view(100, 1, 28, 28))
            
            outputs = model(test)
            
            predictions = torch.max(outputs, 1)[1].to(device)
            predictions_list.append(predictions)
            correct += (predictions == labels).sum()
            
            total += len(labels)
            

    correct=correct.cpu().detach().numpy()


            
    accuracy = correct * 100 / total
    loss_list.append(loss.cpu().detach().numpy())
    accuracy_list.append(accuracy)

    cost_list.append(cost)





fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(cost_list, color=color)
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('Cost', color=color)
ax1.tick_params(axis='y', color=color)
    
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color) 
ax2.set_xlabel('epoch', color=color)
ax2.plot( accuracy_list, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()


plt.show()
