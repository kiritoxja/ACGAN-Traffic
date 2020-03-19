import pandas as pd
from imblearn.over_sampling import SMOTE # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler # 欠抽样处理库RandomUnderSampler
from sklearn.svm import SVC # SVM中的分类算法SVC
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler

from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

# 导入数据文件

from sklearn.datasets import make_classification
from collections import Counter
X, y = make_classification(n_samples=1000, n_features=3, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[ 0.06, 0.94],
                           class_sep=0.8, random_state=0)
print(sorted(Counter(y).items()))
print(type(X))
print(y)
ada = ADASYN(random_state=42)
X_res, y_res = ada.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))