import os
import csv
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyData(Dataset):

    # root_dir: 图片的文件夹
    # csv_path: id 和 label 的路径
    # 模型的输入的图片尺寸可以作为参数来指定
    # image to Tensor 的顺序需要放在靠后面的位置
    def __init__(self, root_dir, subSet):
        self.root_dir = root_dir
        self.subSet   = subSet
        self.csvFile  = self.subSet + "Data.csv"
        self.csv_path = os.path.join(self.root_dir, self.csvFile)
        self.label_dict =  {1 : "landslide", 0 : "ground"}
        self.images, self.labels = self.load_csv(self.csv_path)
        self.transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
   
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        label_id = self.labels[index]
        id = self.label_dict[label_id]
        img_path = os.path.join(self.root_dir, self.subSet, id, self.images[index])
        image = Image.open(img_path)
        img   = self.transforms(image)
        label = self.labels[index]
        label = torch.tensor(label)
        return img, label

    def load_csv(self, csvPath):
            images, labels = [], []
            with open(csvPath) as f:
                reader = csv.reader(f)
                for row in reader:
                    img, label = row  
                    label = int(label)
                    images.append(img)
                    labels.append(label)
            assert len(images) == len(labels)  # 保证数据一致
            return images, labels


# root_dir = "E:/分类模型数据集/二分类数据集/part/"

# trainData = MyData(root_dir, "train")

# print("训练数据集的样本容量: {}".format(len(trainData))) 
# # print(trainData)

# train_loader = DataLoader(dataset = trainData, shuffle=True, batch_size=4, num_workers=0)
# img, target = train_loader.dataset[3]
# print(img.shape, target)
# print(type(target))