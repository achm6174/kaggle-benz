import pandas as pd

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer


train_original = pd.read_csv('data/train.csv')
test_original = pd.read_csv('data/test.csv')
for k in train_original.keys():
    if len(train_original[k].unique()) == 1:
        train_original.pop(k)
        test_original.pop(k)

df_stack_index = pd.read_csv("data/stack_index.csv")
assert all(df_stack_index.ID == train_original.ID)
train_original = pd.merge(train_original, df_stack_index, on='ID')

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

    target = train.pop('y')
    train.pop('ID')
    test_ids = test.pop('ID')

    for k in train.keys():
        if len(train[k].unique()) == 1:
            train.pop(k)
            test.pop(k)



    train, test = map(lambda df: [row for _, row in df.iterrows()],
                      (train, test))

    dv = DictVectorizer()
    sfs = SFS(LinearRegression(),
              k_features=1,
              forward=True,
              floating=False,
              verbose=2,
              scoring='r2',
              cv=3)

    pipe = make_pipeline(dv, sfs, LinearRegression())
    pipe.fit(train, target)
    preds = pipe.predict(test)
    res = pd.DataFrame({'ID': test_ids, 'y': preds})
    save_df.append(res)


df_result = pd.concat(save_df)
assert len(df_result.ID) == len(train_original.ID)
df_result.sort_values("ID", inplace=True)
df_result.reset_index(inplace=True, drop=True)
assert all(df_result.ID == train_original.ID)
df_result.to_csv('level_1/cache/train/main_2.csv', index=False)
