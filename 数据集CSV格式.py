import numpy as np
import os
import pandas as pd
import csv


"""
    将数据的名称存到 CSV文件中
"""

def saveCSV(model, class_dir):
    data_path = os.path.join(root_dir, model, class_dir)
    img_ids = os.listdir(data_path)
    res = []
    for id in img_ids:
        tmp = []
        tmp.append(id)
        tmp.extend(np.array([  
            Classes[class_dir]
        ]))
        res.append(tmp)
    return res

root_dir = "E:/分类模型数据集/二分类数据集/part/"
subSet = ["train", "test"]
Classes = {"landslide": 1,
           "ground": 0}
Class_dir = ["landslide", "ground"]

for sub in subSet:
    data = []
    for label in Classes:
        data.extend(saveCSV(sub, label))
    save_path = root_dir + sub + "Data.csv"
    csvFile = open(save_path, "w+", newline='')
    try:
        writer = csv.writer(csvFile)
        for i in range(len(data)):
            writer.writerow(data[i])
    finally:
        csvFile.close()


