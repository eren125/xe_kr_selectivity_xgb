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
df_data = pd.read_csv('data/all_columns.csv',index_col=0)
X_data, y_data, group = df_data.iloc[:,3:], df_data['G_2080'], df_data["unique_chemcomp"]

df_train = pd.read_csv('data/train.csv',index_col=0)
X_train, y_train, group_train = df_train.iloc[:,3:], df_train['G_2080'], df_train["unique_chemcomp"]

df_test = pd.read_csv('data/test.csv',index_col=0)
X_test, y_test, group_test = df_test.iloc[:,3:], df_test['G_2080'], df_test["unique_chemcomp"]

feature_names = X_train.columns

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

X_shap = df_data[feature_names]
shap_values = explainerModel.shap_values(X_shap.to_numpy())

resultX = pd.DataFrame(shap_values, columns = feature_names)
vals = np.abs(resultX.values).mean(0)

shap_importance = pd.DataFrame(list(zip(feature_names, vals)),columns=['features name','feature_importance_vals']).sort_values(by=['feature_importance_vals'],ascending=True)

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
top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))

# make SHAP plots of the three most important features
for i in range(30):
    if feature_names[top_inds[i]]=="G_0":
        interaction_index="auto"
    else :
        interaction_index="G_0"
    
    if feature_names[top_inds[i]] in ["delta_pore"]:
        shap.dependence_plot(top_inds[i], shap_values, X_shap.to_numpy(),feature_names=feature_names,show=False, xmin=0.003, interaction_index="G_0")
        plt.xscale('log')
    elif feature_names[top_inds[i]]=="pore_dist_neff":
        shap.dependence_plot(top_inds[i], shap_values, X_shap.to_numpy(),feature_names=feature_names,show=False, xmin=0.8, interaction_index="G_0")
        plt.xscale('log')
    else:
        shap.dependence_plot(top_inds[i], shap_values, X_shap.to_numpy(),feature_names=feature_names, interaction_index=interaction_index, show=False)

    plt.savefig('PDP/%s.pdf'%feature_names[top_inds[i]].replace('/','_').replace('%','_percent'), dpi=240, bbox_inches='tight')
    plt.show()

# %%
df_test["AE"] = abs(df_test['pred']-df_test['true'])
data = df_test.sort_values(by="AE", ascending=False)
data[(abs(data["AE"])<1)&(data["true"]<-6)].head()
data[(abs(data["AE"])>1)].head()

# %%
data_test = pd.merge(df_data.reset_index(),df_test.reset_index(), on="index",how="right")

data_test["AE"] = abs(data_test['pred']-data_test['true'])
data_test["DR"] = abs(data_test['G_0']-data_test['true'])

data_test[(abs(data_test["AE"])<0.5)&(abs(data_test["DR"])>1)&(data_test["true"]<-6)].head(20)

# %% selectivity drop
n_test = 22
print(df_test.loc[n_test])
X_outlier = X_test.loc[n_test:,:]

shap.plots.waterfall(explainerModel(X_outlier)[0], max_display=11, show=False)
plt.savefig("plot/%s.pdf"%df_data.loc[n_test,"Structures"], dpi=240, bbox_inches='tight')
plt.show()
print(df_data.loc[n_test,"Structures"])
print(df_data.loc[n_test,"G_2080"])
print(df_data.loc[n_test,"G_0"])
print(df_data.loc[n_test,"chan_mean_dim"])
print(df_test.loc[n_test,"pred"])

# %%
shap.plots.waterfall(explainerModel(X_outlier)[0], max_display=len(feature_names),
                      show=False
                      )
plt.savefig("plot/%s_all.pdf"%df_data.loc[n_test,"Structures"], dpi=240, bbox_inches='tight')
plt.show()

# %%
data_test["s0"] = np.exp(-data_test['G_0']/(R*T))
data_test[(abs(data_test["AE"])<0.5)&(abs(data_test["DR"])<0.1)&(data_test["s0"]>10)].sort_values(by="s0", ascending=False).head(20)

# %% no selec drop
n_test = 3003

df_test.loc[n_test]
X_outlier = X_test.loc[n_test:,:]

shap.plots.waterfall(explainerModel(X_outlier)[0], max_display=11,
                      show=False
                      )
plt.savefig("plot/%s.pdf"%df_data.loc[n_test,"Structures"], dpi=240, bbox_inches='tight')
plt.show()
print(df_data.loc[n_test,"Structures"])
print(df_data.loc[n_test,"G_2080"])
print(df_data.loc[n_test,"G_0"])
print(df_data.loc[n_test,"chan_mean_dim"])
print(df_test.loc[n_test,"pred"])

# %%
shap.plots.waterfall(explainerModel(X_outlier)[0], max_display=len(feature_names),
                      show=False
                      )
plt.savefig("plot/%s_all.pdf"%df_data.loc[n_test,"Structures"], dpi=240, bbox_inches='tight')
plt.show()

# %%
