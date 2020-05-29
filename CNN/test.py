import numpy as np
import pickle
import pandas

with open(r"E:\MyProject\ACGAN-Traffic\CNN\DateSet\ByteLengthSet\2209_augmentTrain.pkl","rb") as f:
    data1 = pickle.load(f)
    print(data1)
    with open(r"E:\MyProject\ACGAN-Traffic\CNN\DateSet\ByteLengthSet\2209_train.pkl","rb") as f:
        data2 = pickle.load(f)
        print(data2)
        data3 = pandas.concat([data1,data2])
        data3.info()
        with open(r'E:\MyProject\ACGAN-Traffic\CNN\DateSet\ByteLengthSet\2209_final_train.pkl', 'wb') as writefile:
            pickle.dump(data3, writefile)