# %%
import pandas as pd 
import numpy as np 
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import r2_score
import pickle

def rmse(a,b):
    return np.sqrt( np.mean ((a-b)**2) )
def mae(a,b):
    return np.mean( abs( a-b ) )

random_state = 123
T = 298.0
R = 8.31446261815324e-3

# %%
# import training data
df_data = pd.read_csv('data/train.csv',index_col=0)
X_data, y_data, group = df_data.iloc[:,3:], df_data['G_2080'], df_data["unique_chemcomp"]

df_train = pd.read_csv('data/train.csv',index_col=0)
X_train, y_train, group_train = df_train.iloc[:,3:], df_train['G_2080'], df_train["unique_chemcomp"]

df_test = pd.read_csv('data/test.csv',index_col=0)
X_test, y_test, group_test = df_test.iloc[:,3:], df_test['G_2080'], df_test["unique_chemcomp"]

# %%
# Load ML model
filename = 'model/final_selectivity_model.sav'
xgb_reg = pickle.load(open(filename, 'rb'))

y_pred_train = xgb_reg.predict(X_train.to_numpy())
df_train = pd.DataFrame(data={'pred':y_pred_train, 'true':y_train.to_numpy()},index=y_train.index).dropna()

y_pred_test = xgb_reg.predict(X_test.to_numpy())
df_test = pd.DataFrame(data={'pred':y_pred_test, 'true':y_test.to_numpy()},index=y_test.index).dropna()

# %%
fig, ax = plt.subplots(figsize=(12,14))
xgb.plot_importance(xgb_reg,ax=ax)
plt.show()

# %%
# EXPLAINABILITY
explainerModel = shap.TreeExplainer(xgb_reg)

X_shap = X_train
X_shap = X_test[X_test['G_0']<0]
# X_shap = df_data[abs(df_data['G_0']-df_data['G_2080'])>2][X_columns]
X_shap = X_data
shap_values = explainerModel.shap_values(X_shap.to_numpy())

feature_names = X_shap.columns
resultX = pd.DataFrame(shap_values, columns = feature_names)
vals = np.abs(resultX.values).mean(0)

shap_importance = pd.DataFrame(list(zip(feature_names, vals)),columns=['features name','feature_importance_vals']).sort_values(by=['feature_importance_vals'],ascending=True)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(12,14))
shap_importance[shap_importance["features name"]!="G_0"].plot.barh(x="features name", y="feature_importance_vals", ax=ax)
ax.set(xlabel="mean(|SHAP value|) (average impact on model output magnitude)", ylabel="Features")
ax.get_legend().remove()
fig.savefig('plot/Feature_importance_shapbased_zoom.pdf', dpi=240,bbox_inches = 'tight')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(12,14))
shap_importance.plot.barh(x="features name", y="feature_importance_vals", ax=ax)
ax.set(xlabel="mean(|SHAP value|) (average impact on model output magnitude)", ylabel="Features")
ax.get_legend().remove()
fig.savefig('plot/Feature_importance_shapbased.pdf', dpi=240,bbox_inches = 'tight')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(12,10))
shap_importance.sort_values(by="feature_importance_vals",ascending=False).reset_index().iloc[:18].sort_values(by="feature_importance_vals",ascending=True).plot.barh(x="features name", y="feature_importance_vals", ax=ax)
ax.set(xlabel="mean(|SHAP value|) (average impact on model output magnitude)", ylabel="Features")
ax.get_legend().remove()
fig.savefig('plot/Feature_importance_shapbased_18top.pdf', dpi=240,bbox_inches = 'tight')
plt.show() 

# %%
plt.figure(figsize=(18,18))
cor = df_data.iloc[:,2:].corr()
sns.heatmap(cor, annot=True,fmt=".2f", cmap=plt.cm.seismic)
plt.savefig('plot/Feature_restrained_correlation.pdf', dpi=240,bbox_inches = 'tight')
plt.show()

# %%
plt.rcParams.update({'font.size': 10})
data = df_data[["G_2080","G_0"]]
data['G_rd'] =  abs(data['G_0'] - data['G_2080'])

x = data["G_2080"]
y = data["G_0"]
z = data['G_rd']

cmap = sns.color_palette("flare", as_cmap=True)
f, ax = plt.subplots()
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", alpha=0.1, color = "gray")
points = ax.scatter(x, y, c=z, s=2, alpha=0.8, cmap=cmap)
lim_min = -15
lim_max = 15
ax.set_xlim(left=lim_min,right=lim_max)
ax.set_ylim(bottom=lim_min,top=lim_max)
plt.xlabel(r"Gibbs free energy of exchange at infinite dilution $\Delta G_0$ [kJ/mol]")
plt.ylabel(r"Gibbs free energy of exchange at 1 bar $\Delta G_1$ [kJ/mol]")
ax.set_aspect('equal', adjustable='box')
clb = f.colorbar(points)
clb.ax.set_title(r"d$_r$($\Delta G_0$,$\Delta G_1$)[kJ/mol]",fontsize=8)
plt.savefig('plot/Scatterplot_G1_G0.pdf', dpi=480)
plt.show() 

# %%
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
s = sns.scatterplot(y='G_0', x='G_2080', data=df_data, s=10, alpha=0.7, ax=ax, label="Whole set (%d structures)"%len(df_data))
ax.set(ylabel=r"Gibbs free energy of exchange at infinite dilution G$_0$ [kJ/mol]", 
xlabel=r"Gibbs free energy of exchange at 1 bar G$_1$ [kJ/mol]")
ax.set_aspect('equal', adjustable='box')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color = "gray")
lim_min = -14
lim_max = 8
plt.xlim(left=lim_min,right=lim_max)
plt.ylim(bottom=lim_min,top=lim_max)
plt.show() 

# %%
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
s = sns.scatterplot(x='true', y='pred', data=df_train, s=10, alpha=0.7, ax=ax, label="Training set (%d structures)"%len(df_train))
s = sns.scatterplot(x='true', y='pred', data=df_test, s=10, alpha=0.8, ax=ax, label="Test set (%d structures)"%len(df_test))
ax.set(xlabel=r"True Gibbs free energy of exchange (1 bar) G$_1$ [kJ/mol]",
ylabel=r"ML Predicted Gibbs free energy of exchange G$_1$ [kJ/mol]")
ax.set_aspect('equal', adjustable='box')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color = "gray")
# lim_min = -14
# lim_max = 8
lim_min = -15
lim_max = 15
plt.xlim(left=lim_min,right=lim_max)
plt.ylim(bottom=lim_min,top=lim_max)
plt.savefig('plot/Scatterplot_G1_prediction.pdf', dpi=480)
plt.show() 

# %%
plt.rcParams.update({'font.size': 14})
df_train["s1_true"] = np.exp(-df_train["true"]/(R*T))
df_train["s1_pred"] = np.exp(-df_train["pred"]/(R*T))
df_test["s1_true"] = np.exp(-df_test["true"]/(R*T))
df_test["s1_pred"] = np.exp(-df_test["pred"]/(R*T))
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
s = sns.scatterplot(x='s1_true', y='s1_pred', data=df_train, s=10, alpha=0.7, ax=ax, label="Training set (%d structures)"%len(df_train))
s = sns.scatterplot(x='s1_true', y='s1_pred', data=df_test, s=10, alpha=0.8, ax=ax, label="Test set (%d structures)"%len(df_test))
ax.set(xlabel=r"True ambient-pressure selectivity $s_1$",
ylabel=r"ML Predicted ambient-pressure selectivity $s_1$")
ax.set_aspect('equal', adjustable='box')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color = "gray")
lim_min = -0.001
lim_max = 250
plt.xlim(left=lim_min,right=lim_max)
plt.ylim(bottom=lim_min,top=lim_max)
plt.savefig('plot/Scatterplot_S1_prediction.pdf', dpi=240)
plt.show() 

rmse_test = rmse(np.log10(df_test['s1_pred']),np.log10(df_test['s1_true']))
print("log10-RMSE test:%s"%rmse_test)
mae_test = mae(np.log10(df_test['s1_pred']),np.log10(df_test['s1_true']))
print("log10-MAE test:%s"%mae_test)
r2_log_test = r2_score(np.log10(df_test['s1_pred']),np.log10(df_test['s1_true']))
print("R2 score on log test:%s"%r2_log_test)
rmse_test = rmse(df_test['s1_pred'],df_test['s1_true'])
print("RMSE test:%s"%rmse_test)
mae_test = mae(df_test['s1_pred'],df_test['s1_true'])
print("MAE test:%s"%mae_test)
# %%
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
s = sns.scatterplot(x='s1_true', y='s1_pred', data=df_train, s=10, alpha=0.7, ax=ax, label="Training set (%d structures)"%len(df_train))
s = sns.scatterplot(x='s1_true', y='s1_pred', data=df_test, s=10, alpha=0.8, ax=ax, label="Test set (%d structures)"%len(df_test))
ax.set(xlabel=r"True ambient-pressure selectivity $s_1$",
ylabel=r"ML Predicted ambient-pressure selectivity $s_1$")
ax.set_aspect('equal', adjustable='box')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color = "gray")
plt.xscale('log')
plt.yscale('log')
lim_min = -0.001
lim_max = 250
plt.xlim(left=lim_min,right=lim_max)
plt.ylim(bottom=lim_min,top=lim_max)
plt.savefig('plot/Scatterplot_S1_prediction_logscale.pdf', dpi=240)
plt.show() 

# %%
data = df_data[["G_2080","G_0","delta_VF_18_20"]].sort_values(by="delta_VF_18_20",ascending=False)
x = data["G_0"]
y = data["G_2080"]
z = np.log10(data['delta_VF_18_20'])

cmap = sns.color_palette("rocket_r", as_cmap=True)
f, ax = plt.subplots()
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", alpha=0.4, color = "gray")
points = ax.scatter(x, y, c=z, s=2, alpha=0.8, cmap=cmap)
lim_min = -15
lim_max = 8
ax.set_xlim(left=lim_min,right=lim_max)
ax.set_ylim(bottom=lim_min,top=lim_max)
plt.ylabel(r"Gibbs free energy of exchange"+"\n at infinite dilution G$_0$ [kJ/mol]")
plt.xlabel(r"Gibbs free energy of exchange at 1 bar G$_1$ [kJ/mol]")
ax.set_aspect('equal', adjustable='box')
clb = f.colorbar(points)
clb.ax.set_title(r"log10(Delta_VF)",fontsize=8)
# plt.savefig('plot/D_log-diameter_colored_s.pdf', dpi=240)
plt.show() 

# %%
probe = "delta_VF_18_20"
df_data['delta'] = df_data['G_2080']-df_data['G_0']
data = df_data[["delta",probe,"G_0"]].sort_values(by="G_0",ascending=False)
x = data["delta"]
y = np.log10(data[probe])
z = data['G_0']

cmap = sns.color_palette("rocket_r", as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(x, y, c=z, s=2, alpha=0.8, cmap=cmap)
# lim_min = -15
# lim_max = 8
# ax.set_xlim(left=lim_min,right=lim_max)
# ax.set_ylim(bottom=lim_min,top=lim_max)
plt.ylabel(r"Delta between G$_0$ at 298K and 900K [kJ/mol]")
plt.xlabel(r"delta G$_0$ G$_1$")
clb = f.colorbar(points)
clb.ax.set_title(r"G$_1$",fontsize=8)
plt.show() 

# %%
