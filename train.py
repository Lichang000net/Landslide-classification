from NewData import *
from MyModel import *
import numpy as np
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from utils import *



root_dir = "E:/分类模型数据集/二分类数据集/part/"
train_data = MyData(root_dir, "train")
test_data  = MyData(root_dir, "test")

print("train data size: {}".format(len(train_data)))
print("test  data size: {}".format(len(test_data)))

learning_rate = 1e-3
batch_size = 4
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50

train_dataloader = DataLoader(dataset=train_data,batch_size=batch_size, shuffle=True)
test_dataloader  = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

alexnet = AlexNet()
alexnet.to(device)
optimizer = optim.SGD(alexnet.parameters(), lr=learning_rate)
# optimizer = optim.Adam(alexnet.parameters(), lr=learning_rate)
loss_fn = CrossEntropyLoss()


for  i in range(epochs):
    
    alexnet.train()
    train_loss = 0
    train_preds = []
    truth = []
    for img, labels in train_dataloader:
        img, labels = img.to(device), labels.to(device)
        outputs = alexnet(img)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
        # 为什么用索引最大值？ 索引对应的就是 0ne-hot 的类别
        train_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        truth.extend(labels.cpu().numpy())
    train_cm = confusion_matrix(truth, train_preds)
    train_acc = calculate_all_prediction(train_cm)
        
    alexnet.eval()
    val_loss = 0
    val_preds = []
    target = []
    for data in test_dataloader:
        img, labels = data
        img, labels = img.to(device), labels.to(device)
        outputs = alexnet(img)
        loss = loss_fn(outputs, labels)
        val_loss += loss
        val_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        target.extend(labels.cpu().numpy())
    val_cm = confusion_matrix(target, val_preds)
    val_acc = calculate_all_prediction(val_cm)
    print("Epoch {}  Train: loss: {:.3}  acc: {:.3}  Val: loss: {:.3}  acc: {:.3}".format(i+1, train_loss, train_acc, val_loss, val_acc))


        
