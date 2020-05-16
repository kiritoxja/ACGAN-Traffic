from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from MetaCost import MetaCost
import pickle as pk
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# [(0, 737), (1, 599), (2, 1069), (3, 2347), (5, 1441), (9, 2395), (10, 2456), (11, 2481), (12, 2399), (13, 2390)]

test = pk.load(open("500_test.pkl", "rb"))
np_test = test.values
y_test = np_test[:,0]
# y_test = pk.load(open("y_test.pkl", "rb"))
pk.dump(y_test, open("y_test.pkl", "wb+"))
y_predict = pk.load( open("y_predict.pkl", "rb"))

print(sorted(Counter(y_test).items()))
print(sorted(Counter(y_predict).items()))


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
