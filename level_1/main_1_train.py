import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import copy
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
from sklearn.metrics import r2_score

# read stack index
df_stack_index = pd.read_csv("data/stack_index.csv")
train_original = pd.read_csv('data/train.csv')
test_original = pd.read_csv('data/test.csv')

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


# process columns, apply LabelEncoder to categorical features
for c in train_original.columns:
    if train_original[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train_original[c].values) + list(test_original[c].values))
        train_original[c] = lbl.transform(list(train_original[c].values))
        test_original[c] = lbl.transform(list(test_original[c].values))

assert all(df_stack_index.ID == train_original.ID)
train_original = pd.merge(train_original, df_stack_index, on='ID')

#df_stack_index[df_stack_index.stack_index==1].ID
save_df = []
for i in range(0, max(df_stack_index.stack_index+1)):
    # read datasets
    train = train_original[train_original.stack_index!=i].copy()
    test = train_original[train_original.stack_index==i].copy()
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    del train['stack_index']
    del test['stack_index']
    del test['y']

    # shape
    print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))


    ##Add decomposed components: PCA / ICA etc.
    n_comp = 12

    # tSVD
    tsvd = TruncatedSVD(n_components=n_comp, random_state=6174)
    tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
    tsvd_results_test = tsvd.transform(test)

    # PCA
    pca = PCA(n_components=n_comp, random_state=6174)
    pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
    pca2_results_test = pca.transform(test)

    # ICA
    ica = FastICA(n_components=n_comp, random_state=6174, tol=1)
    ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
    ica2_results_test = ica.transform(test)

    # Append decomposition components to datasets
    for i in range(1, n_comp+1):
        train['pca_' + str(i)] = pca2_results_train[:,i-1]
        test['pca_' + str(i)] = pca2_results_test[:, i-1]

        train['ica_' + str(i)] = ica2_results_train[:,i-1]
        test['ica_' + str(i)] = ica2_results_test[:, i-1]

    #    train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
    #    test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]

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
    print best_iteration
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
    save_df.append(output)

df_result = pd.concat(save_df)
assert len(df_result.ID) == len(train_original.ID)
df_result.sort_values("ID", inplace=True)
df_result.reset_index(inplace=True, drop=True)
assert all(df_result.ID == train_original.ID)
df_result.to_csv('level_1/cache/train/main_1.csv', index=False)
