# encoding=utf-8
import pandas as pd
import numpy as np

df = pd.read_csv("./data/HR_.csv")

l_s = df["left"]

print(l_s.value_counts())