#!/usr/bin/env python
# coding: utf-8

# ## Capstone project Alex Khvatov

import pickle

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pathlib import Path

pd.options.mode.copy_on_write = True

#Parameters

#output_file = "model.bin"

columns = [
    'quality',
    'pre_screening',
    'ma1',
    'ma2',
    'ma3',
    'ma4',
    'ma5',
    'ma6',
    'exudate1',
    'exudate2',
    'exudate3',
    'exudate4',
    'exudate5',
    'exudate6',
    'exudate7',
    'exudate8',
    'macula_opticdisc_distance',
    'opticdisc_diameter',
    'am_fm_classification',
    'class'
]

path_to_data = Path.resolve(Path("./data/messidor_features.arff"))
path_to_model = Path.resolve(Path("./dist/model.bin"))
df = pd.read_csv(path_to_data, skiprows=24, names=columns)


numeric_columns = ['ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6',
       'exudate1', 'exudate2', 'exudate3', 'exudate4', 'exudate5', 'exudate6',
       'exudate7', 'exudate8', 'macula_opticdisc_distance',
       'opticdisc_diameter']


categorical_columns = ['quality', 'pre_screening', 'am_fm_classification']
for c in categorical_columns:
    df[c] = df[c].astype(str)

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=1)

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

y_train = df_train['class'].values
y_val = df_val['class'].values
y_test = df_test['class'].values

del df_train['class']
del df_val['class']
del df_test['class']

train_dicts=df_train[categorical_columns + numeric_columns].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical_columns + numeric_columns].to_dict(orient='records')
X_val = dv.transform(val_dicts)

def train(df_train, y_train, C=1.0):
    """Trains Logistic Regression model

    Args:
        df_train (_type_): training dataset
        y_train (_type_): training target values
        C (float, optional): Inverse of regularization strength. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    dicts = df_train[categorical_columns + numeric_columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train=dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X_train, y_train)
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical_columns + numeric_columns].to_dict(orient = "records")
    X=dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred

dv, logistic_regression_model = train(df_full_train, df_full_train['class'].values, C=1.0)

y_pred = predict(df_test, dv, logistic_regression_model)

auc = roc_auc_score(y_test, y_pred)

print(f"auc of the final model={auc:.3f}")

# ### Save the model


with open(path_to_model, 'wb') as f_out:
    pickle.dump((dv, logistic_regression_model), f_out)
    
print(f"Model saved to {path_to_model}")
