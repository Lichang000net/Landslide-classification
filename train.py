from NewData import *
from MyModel import *
import numpy as np
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader



root_dir = "E:/分类模型数据集/二分类数据集/part/"
train_data = MyData(root_dir, "train")
test_data  = MyData(root_dir, "test")

print("train data size: {}".format(len(train_data)))
print("test  data size: {}".format(len(test_data)))

learning_rate = 1e-2
batch_size = 2
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50

train_dataloader = DataLoader(dataset=train_data,batch_size=batch_size, shuffle=True)
test_dataloader  = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

alexnet = AlexNet()
alexnet.to(device)
optimizer = optim.SGD(alexnet.parameters(), lr=learning_rate)
loss_fn = CrossEntropyLoss()


for  i in range(epochs):
    alexnet.train()
    train_loss = 0
    for img, labels in train_dataloader:
        img, labels = img.to(device), labels.to(device)
        outputs = alexnet(img)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
    
    alexnet.eval()
    val_loss = 0
    for data in test_dataloader:
        img, labels = data
        img, labels = img.to(device), labels.to(device)
        outputs = alexnet(img)
        loss = loss_fn(outputs, labels)
        val_loss += loss
    print("Epoch {}  Train: loss: {:.5}   Val: loss: {:.5}".format(i+1, train_loss, val_loss))
        
