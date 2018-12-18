import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

#归一化，reshape(-1,1)表示不知道有多少行，所以用-1
print("归一化\n",MinMaxScaler().fit_transform(np.array([1,4,10,15,21]).reshape(-1,1)))

print("标准化\n",StandardScaler().fit_transform(np.array([1,1,1,1,0,0,0,0]).reshape(-1,1)))
