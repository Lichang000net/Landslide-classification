# Landslide-classification
滑坡遥感影像二分类

数据集的格式为:
----根目录:
      train:
         landslide
         ground
      val:
         landslide
         ground
首先按照这样的目录形式创建文件夹，保存数据，然后使用  数据集CSV格式.py 来制作 csv 文件
最后修改模型结构，添加 混淆矩阵，warmUp，优化器等训练技巧。
