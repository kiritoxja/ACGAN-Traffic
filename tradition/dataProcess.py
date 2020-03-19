import csv
import os
import pickle

X_min = []
Y_min = []
X_maj = []
Y_maj = []
X = []
Y = []
dirpath = r"C:\Users\akane\Documents\work\ACGAN-Traffic\tradition\data"
for root,dirs,files in os.walk(dirpath):
    for file in files:
        if file == "Truncate.csv" :
                # 获取文件路径
            filepath = os.path.join(root,file)
            print(filepath)
            tag = filepath.split('\\')[-2]
            print(tag)
            belong = filepath.split('\\')[-3]
            print(belong)

            with open(filepath,newline='',encoding='UTF-8') as csvfile:
                rows = csv.reader(csvfile)
                for row in rows:
                    print(row)
                    X.append(row)
                    x = row

                    if belong == "majority":
                        X_maj.append(row)
                        Y_maj.append(tag)
                    else:
                        X_min.append(row)
                        Y_min.append(tag)

                    X.append(row)
                    Y.append(tag)

temp_X = []
for i in range(0,4000):
    for j in range(0,500):
        temp_X.append(0)
    X_min.append(temp_X)
    Y_min.append("none")
    X_maj.append(temp_X)
    Y_maj.append("none")
    X.append(temp_X)
    Y.append("none")
    print(i)


pickle.dump(X_maj, open( "x_majority.pkl","wb+"))
pickle.dump(Y_maj, open("y_majority.pkl", "wb+"))
pickle.dump(X_min, open("x_minority.pkl", "wb+"))
pickle.dump(Y_min, open("y_minority.pkl", "wb+"))
pickle.dump(X, open("x.pkl", "wb+"))
pickle.dump(Y, open("y.pkl", "wb+"))

