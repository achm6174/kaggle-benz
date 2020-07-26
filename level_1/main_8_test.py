import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import xgboost as xgb

# read datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape
test['y'] = np.nan
test = test[train.columns]
train_test = pd.concat([train, test])
train_test.sort_values("ID", inplace=True)
train_test.reset_index(inplace=True, drop=True)

train_test['previous_1'] = train_test['y'].ffill()
train_test.loc[pd.notnull(train_test['y']), 'previous_1']= np.nan
train_test['next_1'] = train_test['y'].bfill()
train_test.loc[pd.notnull(train_test['y']), 'next_1']= np.nan

train = train_test.loc[pd.notnull(train_test['y']), :].copy()
test = train_test.loc[pd.isnull(train_test['y']), :].copy()

del test['y']
train['previous_1'] = train['y'].shift(1)
train['next_1'] = train['y'].shift(-1)
train.fillna(np.mean(train['y']), inplace=True)
train.reset_index(drop=True, inplace=True)


y_train = train["y"]
y_mean = np.mean(y_train)


# prepare dict of params for xgboost to run with
xgb_params = {
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'base_score': y_mean, # base prediction = mean(target)
    'seed': 6174,
    'silent': 1,
    'eval_metric': 'mae'
}


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'rmse', r2_score(labels, preds)

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params,
                  dtrain,
                  feval=xgb_r2_score,
                  num_boost_round=10000, # increase to have better results (~700)
                  early_stopping_rounds=100,
                  verbose_eval=50,
                  show_stdv=False,
                  maximize=True
                 )
best_iteration = cv_result.shape[0] - 1
print(best_iteration)
cv_mean = cv_result.iloc[-1, 2]
cv_std = cv_result.iloc[-1, 3]
print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))
#num_boost_rounds = len(cv_result)
#print('num_boost_rounds=' + str(num_boost_rounds))

# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=best_iteration)


# check f2-score (to get higher score - increase num_boost_round in previous cell)

print(r2_score(model.predict(dtrain), dtrain.get_label()))

# make predictions and save results
y_pred = model.predict(dtest)

output = pd.DataFrame({'ID': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('level_1/cache/test/main_8.csv', index=False)
