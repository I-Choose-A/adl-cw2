# 测试方法：

首先先下载数据集：

```bash
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
```

然后解压：
```bash
mkdir -p images && tar -xzf images.tar.gz -C images --strip-components=1
```

对于每个task，先进入该task的工作目录：

例如：
```bash
cd task2
```

然后运行：
```bash
python train.py
```

对于task1的两个模型，运行：
```bash
python model1/train.py
python model2/train.py
```

确保都能跑之后就可以
