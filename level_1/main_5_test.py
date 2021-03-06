import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np # linear algebra
np.random.seed(6174)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import mean_squared_error

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
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))


##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
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

## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = train.drop(["y"], axis=1).shape[1], kernel_initializer = 'he_normal'))
    model.add(PReLU())
    #model.add(Dropout(0.1))
    model.add(Dense(100, kernel_initializer = 'he_normal'))
    model.add(PReLU())
    #model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer = 'he_normal'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return(model)

model = nn_model()
pred = np.array([0. for x in range(len(test))])
num_bags = 5
for i in range(num_bags):
    print("#" + str(i))
    model.fit(train.drop(["y"], axis=1).values,
                    y_train,
                    batch_size=32,
                    epochs = 200,
                    verbose = 0)
    pred += model.predict(test.values)[:,0]
pred = pred/num_bags
output = pd.DataFrame({'ID': test['ID'].astype(np.int32), 'y': pred})
output.to_csv('level_1/cache/test/main_5.csv', index=False)
