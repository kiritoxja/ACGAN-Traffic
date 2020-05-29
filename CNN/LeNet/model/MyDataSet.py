
import numpy as np
import pandas as pd
from PIL import Image
from torch import optim,nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class TrafficDataSet(Dataset):

    def __init__(self, data, transforms = None):
        self.data = data
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.transforms = transforms

    def __getitem__(self, index):
        single_image_label = self.LabelsConvet(self.labels[index])
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(47, 47).astype(float)
        # 把 numpy array 格式的图像转换成灰度 PIL image
        img_as_img = Image.fromarray(img_as_np*255)
        img_as_img = img_as_img.convert('L')
        # img_as_img.show()
        # 将图像转换成 tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
            # 返回图像及其 label
        return (img_as_tensor, single_image_label)

    def LabelsConvet(self, OriginalLabel):
        label_name = {0:0 , 1:1 , 2:2 , 3:3, 4:4, 5:5, 6:6, 7:6, 8:6, 9:7, 10 :8, 11:9, 12:10, 13 :11}
        return label_name[OriginalLabel]

    def __len__(self):
        return len(self.data.index)
