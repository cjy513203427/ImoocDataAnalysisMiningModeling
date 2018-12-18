import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#标签化
print("标签化,两种\n",LabelEncoder().fit_transform(np.array(["Down","Up","Up","Down"]).reshape(-1,1)))

print("标签化，三种\n",LabelEncoder().fit_transform(np.array(["Low","Medium","High","Medium","Low"]).reshape(-1,1)))

lb_encoder = LabelEncoder()
lb_tran_f = lb_encoder.fit_transform(np.array(["Red","Yellow","Blue","Green"]))
print("标签化，四种\n",lb_tran_f)

oht_encoder = OneHotEncoder().fit(lb_tran_f.reshape(-1,1))

print("独热化\n",oht_encoder.transform(lb_encoder.transform
                                    (np.array(["Yellow","Blue","Green","Green","Red"])).reshape(-1,1)).toarray())