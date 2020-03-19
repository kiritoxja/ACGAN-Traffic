import os
import numpy as np
import pandas as pd
import pickle as pk

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

def addTrainAndTest(data:pd.DataFrame,trainData:pd.DataFrame,testData:pd.DataFrame,trainRatio:float,numMax:int,balance:bool,classLabel:int):
    '''
    将data按比例添加到训练和测试集
    :param data: 原始类别
    :param trainData: 要添加的训练集
    :param testData: 要添加的测试集
    :param trainRatio:  训练集比例
    :param numMax:  样本阈值
    :param balance: 是否平衡(true 代表样本总数达到阈值)
    :param classLabel: 样本标签
    :return:
    '''
    #先全部打乱
    data = data.sample(frac=1.0)
    if balance:
        train_idx = int(round(trainRatio * numMax))
        trainTemp = data.iloc[:train_idx]
        testTemp = data.iloc[train_idx: numMax - 1]
    else:
        train_idx = int(round(trainRatio * data.shape[0]))
        trainTemp = data.iloc[:train_idx]
        testTemp = data.iloc[train_idx: ]
    #添加标签值
    trainTemp = addLabel(trainTemp,classLabel)
    testTemp = addLabel(testTemp,classLabel)
    trainData = trainData.append(trainTemp)
    testData = testData.append(testTemp)
    return trainData, testData


baseDir = r'../DateSet/500/500'
numMax = 4000
trainRatio = 0.4
sessionLen = 500
trainData = pd.DataFrame()
augmentTrainData = pd.DataFrame()
testData = pd.DataFrame()
umbalancedList = []
os.chdir(baseDir)
dirList = os.listdir()
# 获取所有的类别
classList = list(filter(lambda i:os.path.isdir(i),dirList))
classNum = len(classList)
maxLabel = classNum - 1
minLabel = 0
#类别和标签的对应关系
class2Label = {}
print('classList: ',classList)

#每一类的truncate如果大于等于4000 就全取truncate  否则全取（如果大于4000仍只取4000）
for className in classList:
    truncateCsvPath = os.path.join('./',className,'Truncate.csv')
    paddingCsvPath = os.path.join('./', className, 'Padding.csv')
    truncateCsv = pd.read_csv(truncateCsvPath, header=None)
    if truncateCsv.shape[0] >= numMax:
        #只用trunvate就足够了 是不需要生成算法的
        label = maxLabel
        class2Label[className] = maxLabel
        maxLabel -= 1
        trainData,testData = addTrainAndTest(truncateCsv,trainData,testData,trainRatio,numMax,True,label)
    elif os.path.exists(paddingCsvPath):
        #存在padding
        paddingCsv = pd.read_csv(paddingCsvPath, header=None)
        if paddingCsv.shape[0] + truncateCsv.shape[0] >= numMax:
            #总数够
            label = maxLabel
            class2Label[className] = maxLabel
            maxLabel -= 1
            tempData = truncateCsv.iloc[:]
            paddingCsv = paddingCsv.sample(frac=1.0)
            tempData = tempData.append(paddingCsv.iloc[ : numMax - truncateCsv.shape[0] - 1])
            trainData, testData = addTrainAndTest(tempData, trainData, testData, trainRatio, numMax, True, label)
        else:
            #总数不够 是需要扩充的样本
            label = minLabel
            class2Label[className] = minLabel
            minLabel += 1
            umbalancedList.append(className)
            tempData = truncateCsv.iloc[:]
            tempData = tempData.append(paddingCsv.iloc[:])
            augmentTrainData, testData = addTrainAndTest(tempData,augmentTrainData, testData, trainRatio, numMax, False, label)
    else:
        #不存在padding 且不平衡
        label = minLabel
        class2Label[className] = minLabel
        minLabel += 1
        umbalancedList.append(className)
        augmentTrainData, testData = addTrainAndTest(truncateCsv, augmentTrainData, testData, trainRatio, numMax, False, label)

#保存
print('class2Label: ', class2Label)
print('umbalancedClass : ',umbalancedList)
saveDir = r'../../../trainTestData'
os.chdir(saveDir)
with open(str(sessionLen)+'_train.pkl','wb') as trainF:
    with open(str(sessionLen)+'_test.pkl','wb') as testF:
        with open(str(sessionLen) + '_augmentTrain.pkl', 'wb') as augTrainF:
            pk.dump(trainData, trainF)
            pk.dump(testData, testF)
            pk.dump(augmentTrainData,augTrainF)










