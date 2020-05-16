from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from collections import Counter
import pandas as pd
import numpy as np
from MetaCost import MetaCost
import pickle as pk
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import svm


def countC ():
    origin = [490, 441, 714, 1565, 2, 962, 28, 4, 7, 1600, 1600, 1600, 1600, 1600]
    # origin = [1.0 for i in range(14)]
    total = 12213
    # total = 14
    p = [[0 for i in range(14)] for i in range(14)]
    for i in range(14):
        for j in range(14):
            if i == j:
                continue;
            else:
                p[i][j] = 1000 * origin[j] / origin[i]
    return p



# 获取np数组
df_minority = pk.load(open("500_augmentTrain.pkl","rb"))
df_majority = pk.load(open("500_train.pkl","rb"))
np_minority = df_minority.values
np_majority = df_majority.values
# print(np_minority)

# Y是tag，X是数据
y = np.append( np_minority[:,0], np_majority[:, 0], axis=0)
X = np.append(np.delete(np_minority, 0, axis = 1), np.delete(np_majority,0,axis=1), axis=0)

'''
# 举个例子
X, y = make_classification(n_classes=14, class_sep=14,
                           weights=[0.1, 0.6, 0.15, 0.01, 0.01, 0.01, 0.01, 0.01 ,0.01, 0.01 ,0.01 ,0.02,0.02,0.02], n_informative=14, n_redundant=0, flip_y=0,
                           n_features=14, n_clusters_per_class=1, n_samples=100, random_state=9)


'''

print('Original dataset shape %s' % Counter(y))
S = pd.DataFrame(X)
S['target'] = y
print(S)
# print(S.describe())

# classifer = LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter=3000)
classifer = svm.SVC(C=0.1,kernel='rbf',gamma=10,decision_function_shape='ovr',probability=True)
C = np.array(countC())
print(C)
metaCost = MetaCost(S, classifer, C);
print("start MetaCost")
model = metaCost.fit('target', 14)
print("MetaCost finished")

'''
LR = LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter=3000)
C = np.array(countC())
print(C)
metaCost = MetaCost(S, LR, C);
print("start MetaCost")
model = metaCost.fit('target', 14)
print("MetaCost finished")
'''

print("start predict...")

test = pk.load(open("500_test.pkl", "rb"))
np_test = test.values
X_test = np.delete(np_test, 0, axis = 1)
y_test = np_test[:,0]

print(y_test)
print(sorted(Counter(y_test).items()))
y_predict = model.predict(X_test)
print(y_predict)
print(sorted(Counter(y_predict).items()))

pk.dump(y_test, open("y_test.pkl", "wb+"))
pk.dump(y_predict,open("y_predict.pkl", "wb+"))


print("ACC")
print(accuracy_score(y_test, y_predict))
print("混淆矩阵")
ConfusionMatrix = confusion_matrix(y_test,y_predict)
print(ConfusionMatrix)   # 混淆矩阵
for family in range(14):
    print("Family {} :".format(family + 1))
    print("-----------------------------")
    TP = ConfusionMatrix[family][family]
    FP = ConfusionMatrix.sum(axis=0)[family] - TP
    FN = ConfusionMatrix[family].sum() - TP
    TN = ConfusionMatrix.sum() - FN - TP - FP
    Precison = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F_Score = (2 * Precison * Recall) / (Precison + Recall)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    print("Precison = {:.4f}      Recall = {:.4f}".format(Precison, Recall))
    print("F-Score = {:.4f}       Acc = {:.4f}".format(F_Score, Acc))

'''

print("精确率")
print(precision_score(y_test,y_predict,average=None))  # 精确率
print("召回率")
print(recall_score(y_test,y_predict, average=None))  # 召回率
print("F-Score")
print(f1_score(y_test,y_predict, average=None))  # F-Score
TP = [0 for i in range(14)]            # acc
real = sorted(Counter(y_test).items())
for i in range(0,len(y_test)):
    if y_test[i] == y_predict[i]:
        TP[y_predict[i]] += 1;
acc = [0.0 for i in range(14)]
for i in range(14):
    acc[i] = TP[i]/real[i][1]

print("acc")
print(acc)
'''

'''
# 输出

output = pd.DataFrame(X_test)
output['target'] = y_predict

print(output)
pk.dump(output,  open("MetaCost_new.pkl", "wb+"))

'''

