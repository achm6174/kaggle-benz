import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import r2_score
import copy

df_train = pd.read_csv('data/train.csv')
df_1 = pd.read_csv("level_1/cache/train/main_1.csv")
df_2 = pd.read_csv("level_1/cache/train/main_2.csv")
df_3 = pd.read_csv("level_1/cache/train/main_3.csv")
df_4 = pd.read_csv("level_1/cache/train/main_4.csv")
df_5 = pd.read_csv("level_1/cache/train/main_5.csv")
df_6 = pd.read_csv("level_1/cache/train/main_6.csv")
df_7 = pd.read_csv("level_1/cache/train/main_7.csv")
df_8 = pd.read_csv("level_1/cache/train/main_8.csv")

print(r2_score(df_train.y.values, (df_1.y.values)))
print(r2_score(df_train.y.values, (df_2.y.values)))
print(r2_score(df_train.y.values, (df_3.y.values)))
print(r2_score(df_train.y.values, (df_4.y.values)))
print(r2_score(df_train.y.values, (df_5.y.values)))
print(r2_score(df_train.y.values, (df_6.y.values)))
print(r2_score(df_train.y.values, (df_7.y.values)))
print(r2_score(df_train.y.values, (df_8.y.values)))

from scipy.optimize import minimize

# ======================== NN optimize ======================== #
def f(coord,args):
    return -r2_score(df_train.y.values,
                    coord[0]*(df_1.y.values) +
                    coord[1]*(df_2.y.values) +
                    coord[2]*(df_3.y.values) +
                    coord[3]*(df_4.y.values) +
                    coord[4]*(df_5.y.values) +
                    coord[5]*(df_6.y.values) +
                    coord[6]*(df_7.y.values) +
                    coord[7]*(df_8.y.values)
                    )


initial_guess = np.array([1/5. for x in range(8)])
bnds = [(0., 1.) for i in range(len(initial_guess))]

coord = minimize(f,initial_guess,args = [],
                              method='SLSQP', bounds=bnds)
print(coord)

print(r2_score(df_train.y.values,
                coord.x[0]*(df_1.y.values) +
                coord.x[1]*(df_2.y.values) +
                coord.x[2]*(df_3.y.values) +
                coord.x[3]*(df_4.y.values) +
                coord.x[4]*(df_5.y.values) +
                coord.x[5]*(df_6.y.values) +
                coord.x[6]*(df_7.y.values) +
                coord.x[7]*(df_8.y.values)
                ))



df_1 = pd.read_csv("level_1/cache/test/main_1.csv")
df_2 = pd.read_csv("level_1/cache/test/main_2.csv")
df_3 = pd.read_csv("level_1/cache/test/main_3.csv")
df_4 = pd.read_csv("level_1/cache/test/main_4.csv")
df_5 = pd.read_csv("level_1/cache/test/main_5.csv")
df_6 = pd.read_csv("level_1/cache/test/main_6.csv")
df_7 = pd.read_csv("level_1/cache/test/main_7.csv")
df_8 = pd.read_csv("level_1/cache/test/main_8.csv")

tmp_test =     (
                coord.x[0]*(df_1.y.values) +
                coord.x[1]*(df_2.y.values) +
                coord.x[2]*(df_3.y.values) +
                coord.x[3]*(df_4.y.values) +
                coord.x[4]*(df_5.y.values) +
                coord.x[5]*(df_6.y.values) +
                coord.x[6]*(df_7.y.values) +
                coord.x[7]*(df_8.y.values)
                )
df_test = df_1.copy()
df_test['y'] = tmp_test

df_test.to_csv("level_2/cache/test/main_6.csv", index=False)
#
# df_1 = pd.read_csv("level_1/cache/train/main_1.csv")
# df_1_test = pd.read_csv("level_1/cache/test/main_1.csv")
# df_1_test['y_test'] = df_1_test['y']
# del df_1_test['y']
# df=df_1.merge(df_1_test, "outer", on='ID')
# df.sort_values("ID", inplace=True)
# df.ffill(inplace=True)
# #df.bfill(inplace=True)
# df.dropna(inplace=True)
# print np.corrcoef(df['y'], df['y_test'])
# print np.corrcoef(df[df.index<4000]['y'], df[df.index<4000]['y_test'])
