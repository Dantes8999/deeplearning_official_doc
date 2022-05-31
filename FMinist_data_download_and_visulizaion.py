#############################第一步 载入官方数据集并查看####################
"""
torch 为Pytorch框架库名

torchvision是pytorch的一个图形库,它服务于PyTorch深度学习框架的,由于构建CV模型。
其中：
torchvision.models: 包含常用的模型结构以及预训练模型，方便直接调用 
torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
torchvision.utils: utils可以理解为工具箱 存储了很多有用的方法。

matplotlib为最常见的Python可视化库
"""
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 

"""
此段落定义训练数据 training_data为训练数据变量名
root  表示数据下载后存放的位置
train 取True为调用数据集中的测试数据
download 选择True则会自动下载数据
transform 选择数据处理方法 此处将数据变换为张量
"""
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()  
)

"""
此段落定义训练数据 test_data为训练数据变量名
root  表示数据下载后存放的位置
train 取False为调用测试数据
download 由于已经下载过此处不会反复下载
transform 选择数据处理方法 此处将数据变换为张量与训练集处理方法相同
"""
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#数据的可视化，为了不影响调试，先注释掉

#建立标签编号与文字内涵的对应关系，存于字典结构
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
#创建画布，并规定画布尺寸
figure = plt.figure(figsize=(8,8))
#确定每行每列摆放图片数
cols, rows = 3, 3
#随机取出9张图片及标签进行观察
for i in range(1, cols*rows + 1):
    #获取图片的标签，这里采取在整个数据集上随机抽签的方式
    #randint函数返回一个填充了随机整数的张量，这里张量下限默认为0，上线为数据集图片数量
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #这里可以看到，traning_data每一个索引下的数据结构是一张图片和一个标签共同组成的
    img, label = training_data[sample_idx]
    #逐一将采样的数据样本和标签填充在画布上
    #.add_subplot（）的参数含义为 总行数，总列数，子图位置
    figure.add_subplot(rows, cols, i)
    #前面定义的字典结构数据在这里起到作用
    #将本是数字的离散标签，对应的便于理解的文字内容
    plt.title(f"No.{sample_idx}:{labels_map[label]}")
    #为了美观，关闭图片的坐标轴显示
    plt.axis("off")
    #原图片维度为（1,28,28）,其中1为通道数，28*28为width&hight
    #imshow()将数值显示为热图，此处.squeeze()将图片通道维度去除
    #camp为选择gray，毕竟已经没有颜色了，灰度显示比较舒服
    plt.imshow(img.squeeze(), cmap="gray")
#显示图片
plt.show()







