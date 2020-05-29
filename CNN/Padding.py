import pickle
import pandas as pd
import numpy as np

with open(r'E:\MyProject\ACGAN-Traffic\CNN\DateSet\UnpaddingSet\Gan_tradition.pkl', 'rb+') as pkl_file:

    data = pickle.load(pkl_file)
    print(data)
    data = data.to_numpy()

    padding = np.zeros((22400,29))
    result = np.concatenate((data,padding),axis=1)

    result = pd.DataFrame(result)
    result[0]=result[0].astype('int')
    result.rename(columns={0:'Label'},inplace=True)
    print(result)
    with open(r'E:\MyProject\ACGAN-Traffic\CNN\DateSet\PaddingSet\Gan_tradition.pkl', 'wb') as writefile:
        pickle.dump(result,writefile)
