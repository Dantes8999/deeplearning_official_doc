#############################补充内容 制作定制化的数据集############################
import os
import pandas as pd
from torchvision.io import read_image 

class CustomImageDataset(Dataset):
    """
    在实例化数据集对象时，__init__函数会运行一次。
    我们初始化包含图像的目录、注释文件和两种转换（在下一节有更详细的介绍）。
    """
    def __init__(self, annotations_file, img_dir, transform=None,\
target_transform=None):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    #len函数帮助我们获取数据集中的样本数量
    def __len__(self):
        return len(self.img_labels)

    """
    getitem函数 帮我们抓取指定idx(index 索引)的图片和标签，基于索引，它可以识别图片所在位置，
    通过read_image方法 把图片转化为可以用于模型读取的张量
    通过img_labels,从csv数据中检索到相应的标签
    最后调用两个transform变换函数 并在一个元组中返回张量图像和相应的标签
    """
    def __getitem__(self, idx):
        #.iloc()为pandas库中方法，.iloc[idx, 0]为在img_labels的csv中抽取idx行的第0列数据
        #根据官方文档，csv的第0列数据为文件名，如"tshirt1.jpg"
        #注意：在比较长的变量名是用的是img指图片，短变量名方法名用的是image
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            img = self.transform(image)
        if self.target_transform:
            laebl = self.target_transform(label)
        return image, label
    





