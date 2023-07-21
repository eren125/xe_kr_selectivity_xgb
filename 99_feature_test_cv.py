# %%
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from xgboost import plot_importance

def rmse(a,b):
    return np.sqrt( np.mean ((a-b)**2) )
def mae(a,b):
    return np.mean( abs( a-b ) )

random_state = 123

# %%
# import training data
df_train = pd.read_csv('data/train_all.csv')

feature_names = [
'G_0', 
'G_Xe_900K',
'G_Kr_900K', 
'G_900K', 
"delta_G0_298_900",
"delta_H0_Xe_298_900",
"delta_TS0_298_900",
"enthalpy_std_xenon",
"enthalpy_std_krypton",
"enthalpy_skew",
# "enthalpy_kurt",
"enthalpy_modality",
"mean_grid_xenon",
"mean_grid_krypton",
"std_grid_xenon",
"std_grid_krypton",
'ASA_m2/cm3_1.2',
"delta_VF_18_20",
'PO_VF_2.0',
'D_i_vdw_uff298', 
'delta_pore',
'D_f_vdw_uff298', 
'pore_dist_mean', # extremely corr to LCD
'pore_dist_std',
'pore_dist_skewness', 
'pore_dist_kurtosis',
"pore_dist_neff",
'pore_dist_modality',
# 'mass_g_mol', 
# 'density_kg_m3',
# 'C%',
# 'H%',
# 'O%',
# 'N%',
# 'chan_mean_dim',
# 'chan_count', # divide by symmetry 
# "delta_VF_12_20",
# 'SA_m2/cm3_1.2',
# 'SA_m2/cm3_1.8', 
# 'SA_m2/cm3_2.0', 
# 'ASA_m2/cm3_1.8', 
# 'ASA_m2/cm3_2.0', 
# 'POA_VF_1.2','POA_VF_1.8','POA_VF_2.0',
# 'PO_VF_1.2',
# 'PO_VF_1.8',
# 'LCD/PLD',
# 'DU_C','DU%',
# 'halogen%','metalloid%','ametal%',
# 'metal%',
# 'N/O',
# 'M/C',
# 'M/O'
]

X_train, y_train, group_train = df_train[feature_names], df_train['G_2080'], df_train["unique_chemcomp"]

# %%
default_params = {
    'objective':'reg:squarederror',
    'max_depth': 6,
    'colsample_bytree':1,
    'colsample_bylevel':1,
    'subsample':1,
    'alpha': 0,
    'lambda':1,
    'learning_rate': 0.3,
}

# %%
num_boost_round = 2000
data_dmatrix = xgb.DMatrix(data=X_train.to_numpy(),label=y_train.to_numpy(), feature_names=feature_names)

cv_results = xgb.cv(dtrain=data_dmatrix, params=default_params, nfold=5,num_boost_round=num_boost_round,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=random_state)

print("Mean CV test-RMSE: %.3f"%cv_results.loc[len(cv_results)-1,"test-rmse-mean"])
print("Std CV test-RMSE: %.3f"%cv_results.loc[len(cv_results)-1,"test-rmse-std"])
print("n_estimators: %d"%cv_results.shape[0])
print("Mean CV train-RMSE: %.3f"%cv_results.loc[len(cv_results)-1,"train-rmse-mean"])

# %% matplotlib inline
fig = plt.figure(constrained_layout=True, figsize=(5, 4))
ax = fig.subplots(1, 1)
cv_results.reset_index().plot(y='train-rmse-mean', x='index', ax=ax)
cv_results.reset_index().plot(y='test-rmse-mean', x='index',ax=ax)
plt.xlabel("Number of boost rounds")
plt.ylabel("Mean RMSE")
plt.show()

# %%