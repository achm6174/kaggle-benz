import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import xgboost as xgb

# read datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

feat_col = train.columns

for i in feat_col:
    if (i == 'y') or (i == 'ID'):
        continue
    test[i] = [x in train[i].values for x in test[i]]

print(np.sum(test.drop('ID', axis=1).apply(np.sum, 1)!=376))
print(np.sum(test.drop('ID', axis=1).apply(np.sum, 1)==376))


train['y'].shift(4)
train = pd.read_csv('data/train.csv')
train['y'].shift(1)
print(np.corrcoef(train['y'][1:-1], train['y'].shift(1)[1:-1]))
print(np.corrcoef(train['y'][2:-2], train['y'].shift(2)[2:-2]))
print(np.corrcoef(train['y'][3:-3], train['y'].shift(3)[3:-3]))
print(np.corrcoef(train['y'][4:-4], train['y'].shift(4)[4:-4]))
print(np.corrcoef(train['y'][5:-5], train['y'].shift(5)[5:-5]))
print(np.corrcoef(train['y'][1:-1], train['y'].shift(-1)[1:-1]))
print(np.corrcoef(train['y'][2:-2], train['y'].shift(-2)[2:-2]))
print(np.corrcoef(train['y'][3:-3], train['y'].shift(-3)[3:-3]))
print(np.corrcoef(train['y'][4:-4], train['y'].shift(4)[4:-4]))
print(np.corrcoef(train['y'][5:-5], train['y'].shift(4)[5:-5]))
