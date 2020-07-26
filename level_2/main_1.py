import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import r2_score
import copy

df_train = pd.read_csv('data/train.csv')
df_1 = pd.read_csv("level_1/cache/train/main_1.csv")
df_2 = pd.read_csv("level_1/cache/train/main_2.csv")

r2_score(df_train.y.values, (df_1.y.values + df_2.y.values)/2.)


from scipy.optimize import minimize

# ======================== NN optimize ======================== #
def f(coord,args):
    pred_1, pred_2 = args
    return -r2_score(df_train.y.values, coord[0]*(df_1.y.values) + coord[1]*(df_2.y.values))


initial_guess = np.array([0.5 for x in range(2)])


res = minimize(f,initial_guess,args = [
                                      df_1.y.values,
                                      df_2.y.values
                                     ]
                              ,method='Nelder-Mead')
print(res)
res.x[0]

tmp = (res.x[0]*(df_1.y.values)  + res.x[1]*(df_2.y.values) )
print(r2_score(df_train.y.values, tmp))


df_1_test = pd.read_csv("level_1/cache/test/main_1.csv")
df_2_test = pd.read_csv("level_1/cache/test/main_2.csv")
tmp_test = (res.x[0]*(df_1_test.y.values)  + res.x[1]*(df_2_test.y.values) )

df_test = df_1_test.copy()
df_test['y'] = tmp_test

df_test.to_csv("level_2/cache/test/main_1.csv", index=False)
