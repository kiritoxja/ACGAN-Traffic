import pickle as pk
from pandas import DataFrame
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler,SMOTE,SVMSMOTE, BorderlineSMOTE

index = ['label']
for i in range(0,500):
    index.append(i)
columns = []
for i in range(0,22400):
    columns.append(i)
print(index)


# 获取np数组
df_minority = pk.load(open("500_augmentTrain.pkl","rb"))
df_majority = pk.load(open("500_train.pkl","rb"))
np_minority = df_minority.values
np_majority = df_majority.values
# print(np_minority)


# Y是tag，X是数据
Y = np.append( np_minority[:,0], np_majority[:, 0], axis=0)
X = np.append(np.delete(np_minority, 0, axis = 1), np.delete(np_majority,0,axis=1), axis=0)
print(sorted(Counter(Y).items()))
print("行  %d  列 %s" %(np.size(X,0),np.size(X,1)))

# ----------------------------------------------------------------
#    RandomOverSampler   
ros = RandomOverSampler(random_state=42)    # 他示例用的就是42，就类似随机种子那种东西
X_res, y_res = ros.fit_resample(X, Y)

print(sorted(Counter(y_res).items()))
df = np.c_[y_res,X_res]
print("行  %d  列 %s" %(np.size(df,0),np.size(df,1)))
pk.dump(DataFrame(df,columns= index , index =list(range(np.size(df,0)))) , open("ROS.pkl", "wb+"))


# ----------------------------------------------------------------
#      SMOTE
sm = SMOTE(random_state=42,k_neighbors=1)   # 俺也不知道为啥设1就不会报错了
X_res, y_res = sm.fit_resample(X, Y)

print(sorted(Counter(y_res).items()))
df = np.c_[y_res,X_res]
print("行  %d  列 %s" %(np.size(df,0),np.size(df,1)))
pk.dump(DataFrame(df,columns= index , index =list(range(np.size(df,0)))) , open("SMOTE.pkl", "wb+"))


# ----------------------------------------------------------------
#       SVMSMOTE
sm = SVMSMOTE(random_state=42,k_neighbors=1)   # 俺也不知道为啥设1就不会报错了
X_res, y_res = sm.fit_resample(X, Y)

print(sorted(Counter(y_res).items()))
df = np.c_[y_res,X_res]
print("行  %d  列 %s" %(np.size(df,0),np.size(df,1)))
pk.dump(DataFrame(df,columns= index , index =list(range(np.size(df,0)))) , open("SVMSMOTE.pkl", "wb+"))


# ----------------------------------------------------------------
#       BorderlineSMOTE
sm = BorderlineSMOTE(random_state=42,k_neighbors=1)   # 俺也不知道为啥设1就不会报错了
X_res, y_res = sm.fit_resample(X, Y)

print(sorted(Counter(y_res).items()))
df = np.c_[y_res,X_res]
print("行  %d  列 %s" %(np.size(df,0),np.size(df,1)))
pk.dump(DataFrame(df,columns= index , index =list(range(np.size(df,0)))),  open("BorderlineSMOTE.pkl", "wb+"))

