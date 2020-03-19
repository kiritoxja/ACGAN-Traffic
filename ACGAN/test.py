# import torch
# save_dir = './save'
#
# # torch.save(a,save_dir + r'\a.pkl')
# print(torch.load(save_dir + '/a.pkl'))

# import pandas as pd
# from numpy import *
# import matplotlib.pyplot as plt
# import matplotlib as mpl
#
# mpl.use('TkAgg')
# ts = pd.Series(random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
# ts = ts.cumsum()
# ts.plot()
# plt.show()
# print(mpl.get_backend())

import numpy as np
from matplotlib import pyplot as plt
#
x = np.arange(1, 11)
y = 2 * x + 5
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x, y)
plt.show()
