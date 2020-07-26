import numpy as np
import pandas as pd
from scipy.stats import rankdata
import os

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

seed = 6174
np.random.seed(seed)
datadir = 'data'
cache_dir = 'data'

Y = df_train['y']

stack_index = rankdata(Y, method='ordinal')
stack_index = stack_index % 20
for i in range(len(stack_index)):
    if stack_index[i] >= 10:
        stack_index[i] = 19 - stack_index[i]

df_train['stack_index'] = stack_index
df_stack_index = df_train.loc[:,['ID', 'stack_index', 'y']]
print(df_stack_index.groupby("stack_index").mean()['y'])
print(df_stack_index.groupby("stack_index").count()['y'])

del df_stack_index['y']

assert df_stack_index.shape[0] ==  df_train.shape[0]
assert all(df_stack_index.ID ==  df_train.ID)

df_stack_index.to_csv(os.path.join(cache_dir,'stack_index.csv'), index=False)
