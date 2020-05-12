# gan生成数据组成训练集

import torch
import os
import pickle as pk
import pandas as pd
import numpy as np
from Gan_And_tradition.trainGan_Step2 import Generator

sessionLen = 500
threadshold = 1600
latent_dim = 100

ganDataSavePath = r'./tempTrainData/'+ str(sessionLen) + 'ganData.pkl'
tempDataSavePath = r'./tempTrainData/'+ str(sessionLen) + 'tempTrainData.pkl'
trainDataPath = r'../trainData/500/Gan_tradition.pkl'
generatorPath = r'./save/generator.pkl'

generator = torch.load(generatorPath)


#不需要GAN生成或者样本数太少用传统算法生成的数据
tempData = pk.load(open(tempDataSavePath,'rb'))

#原始gan数据
ganData = pk.load(open(ganDataSavePath,'rb'))


def nomalize(array:np.array):
    max = array.max()
    func = lambda x : 0 if x < 0 else x/max
    return np.array(list(map(func,array)))

def addLabel(dataframe:pd.DataFrame,label:int):
    '''
    为dataframe开头添加相应的标签值
    :param dataframe:
    :param label:
    :return:
    '''
    # 在开头新增标签列
    col_name = dataframe.columns.tolist()
    col_name.insert(0, 'label')
    dataframe = dataframe.reindex(columns=col_name)
    # 添加标签值
    dataframe['label'] = [label for i in range(dataframe.shape[0])]
    return dataframe

#修改ganData  label值到真实值
#ganlabel重排的结果
TrueLable2GanLabel = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
GanLable2TrueLable = {}
for key,value in TrueLable2GanLabel.items():
    GanLable2TrueLable[value] = key
def lable(df):
    return GanLable2TrueLable[df['label']]
ganData['label'] = ganData.apply(lambda row : lable(row),axis=1)
#每类数据原来有多少条
class2Num = dict(ganData.loc[:,'label'].value_counts())


#开始对每类生成
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
#显示计数器
countIndex = 1
for classLabel in class2Num:
    count = class2Num[classLabel]
    #还需要生成这么多条数据
    z = FloatTensor(np.random.normal(0, 1, (threadshold - count, latent_dim)))
    ganLabel = TrueLable2GanLabel[classLabel]
    labels = LongTensor(np.full(shape=threadshold - count,fill_value=ganLabel))
    sessions = generator(z, labels)
    sessions = sessions.cpu().detach().numpy()
    #归一化每条session
    for i in range(sessions.shape[0]):
        sessions[i] = nomalize(sessions[i])
    df = pd.DataFrame(sessions)
    df = addLabel(df,classLabel)
    #添加生成的数据
    ganData = ganData.append(df)
    print('-----------')
    print(countIndex,'/',len(class2Num.keys()))
    countIndex += 1
tempData = tempData.append(ganData)

with open(trainDataPath,'wb') as f:
    pk.dump(tempData,f)

