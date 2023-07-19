# %%
import pandas as pd 
import numpy as np 
from sklearn.model_selection import GroupShuffleSplit,train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import r2_score
import pickle
from xgboost import plot_importance
import chemparse
import math 

def rmse(a,b):
    return np.sqrt( np.mean ((a-b)**2) )
def mae(a,b):
    return np.mean( abs( a-b ) )

# %%
# import training data
df_train = pd.read_csv('data/train.csv')
X_train, y_train, group = df_train.iloc[:,3:], df_train['G_2080'], df_train["unique_chemcomp"]

df_test = pd.read_csv('data/test.csv')
X_test, y_test, group = df_test.iloc[:,3:], df_test['G_2080'], df_test["unique_chemcomp"]

# %%
