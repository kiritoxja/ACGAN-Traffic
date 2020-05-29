import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from LeNet import LeNet
from MyDataSet import TrafficDataSet
import sys



# ====================== 将控制台信息记录 ========================
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()


    def flush(self):
        self.log.flush()

sys.stdout = Logger('E:\MyProject\ACGAN-Traffic\CNN\Result\\2209_final_train.log', sys.stdout)

# ========================    原始标签        ====================================================
#Label:  {'Benign': 13, 'Botnet ARES': 0, 'DDos': 12, 'DoS-GoldenEye': 11, 'Dos-Hulk': 10,
# 'Dos-Slowhttptest': 1, 'DoS-Slowloris': 2, 'FTP-Patator': 3, 'Infiltration': 4, 'PortScan': 9, 'SSH-Patator': 5,
# 'WebAttack-BruteForce': 6, 'WebAttack-Sql Injection': 7, 'WebAttack-Xss': 8}
# ========================    原始标签        =====================================================


# ========================  标签转换  ====================================

label_name = {'0':0 , '1':1 , '2':2 , '3':3,'4':4,'5':5,'6':6,'7':6,'8':6,'9':7,'10':8,'11':9,'12':10,'13':11}

# ===============================================================

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.01
log_interval = 100
val_interval = 1


# ======================== step 1/5 数据加载 =====================

TrainFile = open(r'E:\MyProject\ACGAN-Traffic\CNN\DateSet\ByteLengthSet\2209_final_train.pkl',"rb+")
TrainDate = pickle.load(TrainFile)
print("============= TrainDate ===============")
TrainDate.info()
print("==================================")
TestFile = open(r'E:\MyProject\ACGAN-Traffic\CNN\DateSet\ByteLengthSet\2209_test.pkl','rb+')
TestDate = pickle.load(TestFile)
print("============= TestDate ===============")
TestDate.info()
print("==================================")
TrainFile.close()
TestFile.close()

# 构建MyDataset实例

transform = transforms.Compose([transforms.ToTensor()])
train_data = TrafficDataSet(TrainDate, transform)
test_data = TrafficDataSet(TestDate,transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
# ============================ step 2/5 模型 ============================



net = LeNet(classes=12)
net.initialize_weights()

# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)                        # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)     # 设置学习率下降策略

# ============================ step 5/5 训练集训练 ============================


for epoch in range(MAX_EPOCH):
    print("-------- start train Epoch [{:0>3}] ---------".format(epoch+1))
    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
    Matrix = np.zeros((12, 12), dtype=int) # !!!!!!混淆矩阵
    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        outputs = net(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels.long())
        loss.backward()

        # update weights
        optimizer.step()
        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        Length = labels.size(0)
        # ======================= 混淆矩阵 ==============================
        for nums in range(Length):
            Matrix[labels[nums - 1].item()][predicted[nums - 1].item()] += 1
        # ======================= 混淆矩阵 ==============================
        correct += (predicted == labels).squeeze().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()

        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch+1, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.
    print(Matrix)

    scheduler.step()  # 更新学习率
    print("-------- over train Epoch [{:0>3}] ---------".format(epoch+1))

    print("-------- starts test Epoch [{:0>3}] ---------".format(epoch+1))
    ConfusionMatrix = np.zeros((12,12),dtype=int)

    if (epoch + 1) % val_interval == 0:
        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                # =========== 打印混淆矩阵 ================
                Length = labels.size(0)
                for nums in range(Length):
                    ConfusionMatrix[labels[nums-1].item()][predicted[nums-1].item()]+=1
                # =========== 打印混淆矩阵 ================
                correct_val += (predicted == labels).squeeze().sum().numpy()

                loss_val += loss.item()


            print("TestSet:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch+1, MAX_EPOCH, j + 1, len(test_loader), loss_val, correct_val / total_val))
            print(ConfusionMatrix)
            # ============ 打印每个类别的信息 =============
            for family in range(12):
                print("Family {} :".format(family+1))
                print("-----------------------------")
                TP = ConfusionMatrix[family][family]
                FP = ConfusionMatrix.sum(axis=0)[family]-TP
                FN = ConfusionMatrix[family].sum()-TP
                TN = ConfusionMatrix.sum()-FN-TP-FP
                Precison = TP/(TP+FP)
                Recall = TP/(TP+FN)
                F_Score = (2*Precison*Recall)/(Precison+Recall)
                Acc = (TP+TN)/(TP+TN+FP+FN)
                print("Precison = {:.4f}      Recall = {:.4f}".format(Precison, Recall)  )
                print("F-Score = {:.4f}       Acc = {:.4f}".format(  F_Score   , Acc     ))



    print("-------- over test Epoch [{:0>3}] ---------".format(epoch+1))







