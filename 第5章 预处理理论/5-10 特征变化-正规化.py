import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
#失败，正规化是对行正规化，不是对列
print("正规化\n",Normalizer(norm="l1").fit_transform(np.array([1,1,3,-1,2]).reshape(-1,1)))

print("L1正规化\n",Normalizer(norm="l1").fit_transform(np.array([[1,1,3,-1,2]])))

print("L2正规化\n",Normalizer(norm="l2").fit_transform(np.array([[1,1,3,-1,2]])))
