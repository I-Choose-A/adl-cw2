# 测试方法：

首先先下载数据集：

```bash
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
```

然后解压：
```bash
mkdir -p images && tar -xzf images.tar.gz -C images --strip-components=1
```

下载注释文件：
```bash
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```

解压注释文件：
```bash
mkdir -p annotations && tar -xzf annotations.tar.gz -C annotations --strip-components=1
```


其次创建一个新的虚拟环境，我们要测试一下到底要用哪些包：
```bash
conda create -n cw2 python=3.12 pip
```

激活环境：
```bash
conda activate cw2
```

安装老师要求的包：
```bash
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cpu
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
