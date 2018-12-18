import numpy as np
import pandas as pd

lst = [6,8,10,15,16,24,25,40,67]
print ("等频分箱/等深分箱\n",pd.qcut(lst,q=3))

print ("带Label的等频分箱/等深分箱\n",pd.qcut(lst,q=3,labels=["low","medium","high"]))
