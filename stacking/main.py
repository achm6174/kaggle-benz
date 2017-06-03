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
stack_index = stack_index % 10
df_train['stack_index'] = stack_index
df_stack_index = df_train.loc[:,['ID', 'stack_index']]
#df_stack_index.groupby("stack_index").mean()['y']


df_stack_index.to_csv(os.path.join(cache_dir,'stack_index.csv'), index=False)
