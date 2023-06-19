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
df_pore = pd.read_csv('zeo/PoreSize_Uff298K_coremof2019.csv')

df_info = pd.read_csv('zeo/Zeoinfo_descriptors_coremof2019.csv')
df_info.replace(np.inf,10, inplace=True)
df_chem =  pd.read_csv('zeo/zeoinfo_coremof2019.csv')[['Structures','chem_comp']]

df_vol_12 = pd.read_csv('zeo/VolumeOccupiable_Uff298K1.2_coremof2019.csv')
df_vol_18 = pd.read_csv('zeo/VolumeOccupiable_Uff298K1.8_coremof2019.csv')
df_vol_20 = pd.read_csv('zeo/VolumeOccupiable_Uff298K2.0_coremof2019.csv').rename(columns={"PO_VF":"PO_VF_2.0","POA_VF":"POA_VF_2.0","PONA_VF":"PONA_VF_2.0"})
df_vol = pd.merge(df_vol_12,df_vol_18, on="Structures",  how="left", suffixes=("_1.2", "_1.8"))
df_vol = pd.merge(df_vol,df_vol_20, on="Structures",  how="left")

df_sa_12 = pd.read_csv('zeo/SurfaceArea_Uff298K1.2_coremof2019.csv')
df_sa_18 = pd.read_csv('zeo/SurfaceArea_Uff298K1.8_coremof2019.csv')
df_sa_20 = pd.read_csv('zeo/SurfaceArea_Uff298K2.0_coremof2019.csv').rename(columns={"ASA_m2/cm3":"ASA_m2/cm3_2.0","NASA_m2/cm3":"NASA_m2/cm3_2.0", "SA_m2/cm3":"SA_m2/cm3_2.0"})
df_sa = pd.merge(df_sa_12,df_sa_18, how="left", on="Structures", suffixes=("_1.2", "_1.8"))
df_sa = pd.merge(df_sa,df_sa_20, on="Structures",  how="left")

df_chan = pd.read_csv('zeo/Channel_Uff298K1.8_coremof2019.csv')

# %%
T = 298.0
R = 8.31446261815324e-3
P_0 = 101300          # Pa
mmol2mol = 1e-3

df_xe = pd.read_csv('Screening_CoReMOF_Dataset.csv')
# df_kr = pd.read_csv('krypton_900K.csv')[['Structure_name','E_surface_B','E_surface_B_900']].rename(columns={'Structures':'Structure_name'})

# df_surf_xe = pd.read_csv('cpp_output_final_2k.csv')
# df_surf_xe['Structure_name'] = df_surf_xe['Structure_name'].str.rsplit('_',1).str[0]
# df_surf_kr = pd.read_csv('cpp_output_final_2k_krypton.csv')
# df_surf_kr['Structure_name'] = df_surf_kr['Structure_name'].str.rsplit('_',1).str[0]

df_grid_xe = pd.read_csv('output_grid_0.12_298K_100_Xe.csv')
df_grid_xe['Structures'] = df_grid_xe['Structure_name'].str.rsplit('_',n=1).str[0]
df_grid_xe.drop(columns=["Structure_name"],inplace=True)
df_grid_kr = pd.read_csv('output_grid_0.12_298K_100_Kr.csv')
df_grid_kr['Structures'] = df_grid_kr['Structure_name'].str.rsplit('_',n=1).str[0]
df_grid_kr.drop(columns=["Structure_name"],inplace=True)
df_psd = pd.read_csv("coremof_poredist_uff298K.csv")

df_900_Xe = pd.read_csv('output_grid_0.12_900K_Xe.csv')
df_900_Xe['Structures'] = df_900_Xe['Structure_name'].str.rsplit('_',n=1).str[0]
df_900_Xe['KH_900K'] = 1e3*R*T*df_900_Xe['Henry_coeff_molkgPa']
df_900_Xe['H_900K'] = df_900_Xe['Enthalpy_grid_kjmol']
df_900_Kr = pd.read_csv('output_grid_0.12_900K_Kr.csv')
df_900_Kr['Structures'] = df_900_Kr['Structure_name'].str.rsplit('_',n=1).str[0]
df_900_Kr['KH_900K'] = 1e3*R*T*df_900_Kr['Henry_coeff_molkgPa']
df_900_Kr['H_900K'] = df_900_Kr['Enthalpy_grid_kjmol']
df_900 = pd.merge(df_900_Xe[['Structures','KH_900K','H_900K']],df_900_Kr[['Structures','KH_900K','H_900K']], on='Structures', how='left', suffixes=("_xenon","_krypton"))

df = pd.merge(df_grid_xe,df_grid_kr, on='Structures', how='left', suffixes=("_xenon","_krypton"))

df = pd.merge(df,df_xe, on='Structures', how='left')

df = pd.merge(df,df_psd, on='Structures', how='left')
df = pd.merge(df,df_info, on='Structures', how='left')
df = pd.merge(df,df_chem, on='Structures', how='left')
df = pd.merge(df,df_pore, on='Structures', how='left')
df = pd.merge(df,df_chan, on='Structures', how='left')
df = pd.merge(df,df_sa, on='Structures', how='left')
df = pd.merge(df,df_vol, on='Structures', how='left')
df = pd.merge(df,df_900, on='Structures', how='left')

# %%
df['K_Xe'] = df['K_Xe_widom']
df['K_Kr'] = df['K_Kr_widom']

df['H_Xe_0'] = df['H_Xe_0_widom']
df['H_Kr_0'] = df['H_Kr_0_widom']

df['Delta_H_0'] = df['H_Xe_0'] - df['H_Kr_0']
# Need for enthalpy of krypton maybe ?

df['s_2080_log'] = np.log10(df['s_2080'])

df['Delta_H_2080'] = df['H_Xe_2080'] - df['H_Kr_2080']

df['s_0'] = df['Henry_coeff_molkgPa_xenon']/df['Henry_coeff_molkgPa_krypton']
df['s_0'] = df['s_0'].replace(0,np.nan)

df['s_2080'] = df['s_2080'].replace(0,np.nan)

df.replace([np.inf,-np.inf],np.nan,inplace=True)
df = df[~(df['DISORDER']=='DISORDER')]
print(df.shape[0])

# %%
#### RESTRICTION on non-radioactive ASR 3D-MOFs 
df_data = df[(df['C%']>0)&(df['metal%']>0)] #MOF
print(df_data.shape[0])

df_data = df_data[(df_data['framework_mean_dim']>2)] #3D
df_data = df_data[(df_data['solvent_removed']==1)] #ASR
df_data = df_data[(df_data['radioactive%']==0)] #nonradioactive
print(df_data.shape[0])

#### RESTRICTION on materials porous enough for xenon
df_data = df_data[df_data['D_i_vdw_uff298']>4]
print(df_data.shape[0])

# %%
df_data["chemcomp_dict"] = df_data['chem_comp'].apply(lambda x: chemparse.parse_formula(x))
df_data['atoms_count'] = df_data["chemcomp_dict"].apply(lambda x: sum(x.values()))

# %%
def unique_chemcomp(chemcomp_dict):
    atomic_symbols = [
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]
    i, s = 0, 0
    chemcomp_values = chemcomp_dict.values()
    pgcd = int(list(chemcomp_values)[0])
    unique_comp = ""
    for val in chemcomp_values:
        pgcd = math.gcd(pgcd,int(val))
    while s<len(chemcomp_dict) or i<len(atomic_symbols):
        comp_value = chemcomp_dict.get(atomic_symbols[i],0)
        if comp_value != 0:
            unique_comp += atomic_symbols[i]
            unique_comp += "%d"%(int(comp_value)//pgcd)
            s += 1
        i += 1
    return unique_comp

df_data['unique_chemcomp'] = df_data["chemcomp_dict"].apply(lambda x: unique_chemcomp(x))

# %%
df_data.rename(columns={'Framework Mass [g/mol]':"mass_g_mol", 'Framework Density [kg/m^3]':'density_kg_m3'}, inplace=True)

df_data['G_2080'] = -R*T*np.log(df_data['s_2080'])
df_data['G_0'] = -R*T*np.log(df_data['s_0'])
df_data['G_0_widom'] = -R*T*np.log(df_data['s_0_widom'])
df_data['pore_dist_modality'] = (df_data['pore_dist_skewness']**2+1)/df_data['pore_dist_kurtosis']
df_data['G_Xe_0'] = -R*T*np.log(R*T*df_data['density_kg_m3'] * df_data['Henry_coeff_molkgPa_xenon'])
df_data['G_Kr_0'] = -R*T*np.log(R*T*df_data['density_kg_m3'] * df_data['Henry_coeff_molkgPa_krypton'])

T_h = 900
df_data['G_Xe_900K'] = -R*T_h*np.log(R*T_h*df_data['density_kg_m3'] * df_data['KH_900K_xenon'])
df_data['G_Kr_900K'] = -R*T_h*np.log(R*T_h*df_data['density_kg_m3'] * df_data['KH_900K_krypton'])
df_data['G_900K'] = df_data['G_Xe_900K'] - df_data['G_Kr_900K']

df_data["delta_G0_298_900"] = df_data['G_900K'] - df_data['G_0']
df_data["delta_H0_Xe_298_900"] =  df_data['H_900K_xenon'] - df_data['Enthalpy_grid_kjmol_xenon']
df_data["delta_H0_Kr_298_900"] = df_data['H_900K_krypton'] - df_data['Enthalpy_grid_kjmol_krypton']

df_data["delta_H0_298_900"] = df_data['delta_H0_Xe_298_900'] - df_data['delta_H0_Kr_298_900'] 
df_data['delta_TS0_298_900'] = df_data['delta_G0_298_900'] - df_data["delta_H0_298_900"]

df_data['N/O'] = (df_data['N%']/df_data['O%']).replace(np.inf,10)
df_data['DU%'] = 100/df_data['atoms_count'] + df_data["C%"] - 0.5*(df_data['H%'] + df_data['halogen%'] - df_data['N%'])
df_data['DU_C'] = df_data['DU%']/df_data['C%']
df_data['DU'] = 1 + df_data['atoms_count'] * ( df_data["C%"] - 0.5*(df_data['H%'] + df_data['halogen%'] - df_data['N%']) )/100

df_data['LCD/PLD'] = df_data['D_i_vdw_uff298']/df_data['D_f_vdw_uff298']

df_data['SA_1.2'] = df_data['ASA_1.2'] + df_data['NASA_1.2']
df_data['delta_SA'] = df_data['SA_m2/cm3_1.8'] - df_data['SA_m2/cm3_2.0']
df_data['delta_VF_12_20'] = df_data['PO_VF_1.2'] - df_data['PO_VF_2.0']
df_data['delta_VF_18_20'] = df_data['PO_VF_1.8'] - df_data['PO_VF_2.0']

df_data['delta_pore'] = df_data['D_i_vdw_uff298'] - df_data['pore_dist_mean']

df_data['enthalpy_modality'] = (df_data['enthalpy_skew']**2+1)/df_data['enthalpy_kurt']

df_data['pore_dist_neff_log10'] = np.log10(df_data['pore_dist_neff'])

X_columns = [
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
'C%',
'O%',
'N%',
]

y_column = ['G_2080']

# %%
# Remove nan values (search for other solutions)
# Maybe more suitable to replace by fillna with median or null values
print(len(df_data))
df_data.dropna(subset=y_column,inplace=True)
print(len(df_data))
df_data.dropna(subset=X_columns,inplace=True)
print(len(df_data))

X, y = df_data[X_columns], df_data[y_column]

# %%
# Split on unique composition (add topology even better)
random_state=123
test_size=0.2

gss = GroupShuffleSplit(n_splits=1, test_size=test_size,random_state=123)

train_idx, test_idx = next(gss.split(X=df_data[X_columns], y=df_data[y_column], groups=df_data["unique_chemcomp"]))

# %%
train_data = df_data.reset_index().iloc[train_idx]
test_data = df_data.reset_index().iloc[test_idx]

X_train = train_data[X_columns]
y_train = train_data[y_column]
X_test = test_data[X_columns]
y_test = test_data[y_column]

# %%
num_boost_round = 2000
params = {
    'objective':'reg:squarederror',
    'max_depth': 6,
    'colsample_bytree':0.85,
    'colsample_bylevel':0.65,
    'subsample':0.7,
    'alpha': 0.4,
    'lambda':1,
    'gamma' :0,
    'learning_rate': 0.04,
}

params = {'objective':'reg:squarederror',
          'subsample': 0.6,
           'n_estimators': 1500, 'max_depth': 7, 'learning_rate': 0.02, 'lambda': 0, 'colsample_bytree': 0.95, 'colsample_bylevel': 0.95, 'alpha': 0.6}

# %%
# params['n_estimators'] = cv_results.shape[0]
params['n_estimators'] = 1500

xgb_reg = xgb.XGBRegressor(**params,random_state=random_state)
xgb_reg.fit(X_train, y_train.to_numpy())

y_pred_train = xgb_reg.predict(X_train.to_numpy())
df_train = pd.DataFrame(data={'pred':y_pred_train, 'true':y_train.to_numpy()[:,0]},index=y_train.index).dropna()

rmse_train = rmse(df_train['pred'],df_train['true'])
print("RMSE train:%s"%rmse_train)

mae_train = mae(df_train['pred'],df_train['true'])
print("MAE train:%s"%mae_train)

y_pred_test = xgb_reg.predict(X_test.to_numpy())
df_test = pd.DataFrame(data={'pred':y_pred_test, 'true':y_test.to_numpy()[:,0]},index=y_test.index).dropna()

# data = df_test[df_test['true']<0]
rmse_test = rmse(df_test['pred'],df_test['true'])
print("RMSE test:%s"%rmse_test)

mae_test = mae(df_test['pred'],df_test['true'])
print("MAE test:%s"%mae_test)

r2_test = r2_score(df_test['pred'],df_test['true'])
print("R2 score on log test:%s"%r2_test)

# %%
#Training plot 
RMSE = []
MAE = []
frac = [0.2,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95]
for q in frac:
    xgb_reg_temp = xgb.XGBRegressor(**params,random_state=random_state)
    X_temp, _, y_temp, _ = train_test_split(X_train, y_train, test_size=1-q, random_state=random_state)
    xgb_reg_temp.fit(X_temp, y_temp.to_numpy())
    y_pred_test = xgb_reg_temp.predict(X_test.to_numpy())
    df_test = pd.DataFrame(data={'pred':y_pred_test, 'true':y_test.to_numpy()[:,0]},index=y_test.index).dropna()
    RMSE.append(rmse(df_test['pred'],df_test['true']))
    MAE.append(mae(df_test['pred'],df_test['true']))

frac.append(1)
RMSE.append(rmse_test)
MAE.append(mae_test)

# %%
plt.plot(frac,RMSE)
plt.xlabel("Fraction of the initial training set")
plt.ylabel("RMSE on the same test size (kJ/mol)")
plt.savefig("training_curve.pdf", dpi=440)
plt.show()

# %%
filename = 'model/20230616_final_selectivity_model.sav'
pickle.dump(xgb_reg, open(filename, 'wb'))

# %%
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(12,14))
plot_importance(xgb_reg,ax=ax)
plt.show()

# %%
# EXPLAINABILITY
explainerModel = shap.TreeExplainer(xgb_reg)

X_shap = X_train
X_shap = X_test[X_test['G_0']<0]
# X_shap = df_data[abs(df_data['G_0']-df_data['G_2080'])>2][X_columns]
X_shap = df_data[X_columns]
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
cor = df_data[y_column+X_columns].corr()
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
plt.xlabel(r"Free Enthalpy of exchange at infinite dilution $\Delta G_0$ [kJ/mol]")
plt.ylabel(r"Free Enthalpy of exchange at 1 bar $\Delta G_1$ [kJ/mol]")
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
ax.set(ylabel=r"Free Enthalpy of exchange at infinite dilution G$_0$ [kJ/mol]", 
xlabel=r"Free Enthalpy of exchange at 1 bar G$_1$ [kJ/mol]")
ax.set_aspect('equal', adjustable='box')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color = "gray")
lim_min = -14
lim_max = 8
plt.xlim(left=lim_min,right=lim_max)
plt.ylim(bottom=lim_min,top=lim_max)
plt.show() 

# %%
plt.rcParams.update({'font.size': 12})
data = df_data[["H_Xe_2080","H_900K_xenon"]]
data['H_rd'] =  abs(data['H_Xe_2080'] - data['H_900K_xenon'])

x = data["H_Xe_2080"]
y = data["H_900K_xenon"]
z = data['H_rd']

cmap = sns.color_palette("flare", as_cmap=True)
f, ax = plt.subplots()
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", alpha=0.1, color = "gray")
points = ax.scatter(x, y, c=z, s=2, alpha=0.8, cmap=cmap)
lim_min = -53
lim_max = -6
ax.set_xlim(left=lim_min,right=lim_max)
ax.set_ylim(bottom=lim_min,top=lim_max)
plt.xlabel(r"Xenon adsorption enthalpy at 1 bar $\Delta H_1$ [kJ/mol]")
plt.ylabel(r"Xenon adsorption enthalpy at" "\ninfinite dilution and 900K $\Delta H_0$ [kJ/mol]")
ax.set_aspect('equal', adjustable='box')
clb = f.colorbar(points)
clb.ax.set_title(r"d$_r$($\Delta H_0$(900K),$\Delta H_1$)[kJ/mol]",fontsize=8)
plt.savefig('plot/Scatterplot_H1_H900K.pdf', dpi=240)
plt.show() 

# %%
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
s = sns.scatterplot(x='H_Xe_2080', y='H_900K_xenon', data=df_data, s=10, alpha=0.7, ax=ax, label="Whole set (%d structures)"%len(df_data))
ax.set(ylabel=r"Xenon adsorption enthalpy at infinite dilution and 900K $\Delta H_0$ [kJ/mol]", 
xlabel=r"Xenon adsorption enthalpy at 1 bar $\Delta H_1$ [kJ/mol]")
ax.set_aspect('equal', adjustable='box')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color = "gray")
lim_min = -53
lim_max = -6
plt.xlim(left=lim_min,right=lim_max)
plt.ylim(bottom=lim_min,top=lim_max)
plt.show() 

# %%
plt.rcParams.update({'font.size': 12})
data = df_data[["H_Xe_2080","Enthalpy_grid_kjmol_xenon"]]
data['H_rd'] =  abs(data['H_Xe_2080'] - data['Enthalpy_grid_kjmol_xenon'])

x = data["H_Xe_2080"]
y = data["Enthalpy_grid_kjmol_xenon"]
z = data['H_rd']

cmap = sns.color_palette("flare", as_cmap=True)
f, ax = plt.subplots()
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", alpha=0.1, color = "gray")
points = ax.scatter(x, y, c=z, s=2, alpha=0.8, cmap=cmap)
lim_min = -53
lim_max = -6
ax.set_xlim(left=lim_min,right=lim_max)
ax.set_ylim(bottom=lim_min,top=lim_max)
plt.xlabel(r"Xenon adsorption enthalpy at 1 bar $\Delta H_1$ [kJ/mol]")
plt.ylabel(r"Xenon adsorption enthalpy at" "\ninfinite dilution and 298K $\Delta H_0$ [kJ/mol]")
ax.set_aspect('equal', adjustable='box')
clb = f.colorbar(points)
clb.ax.set_title(r"d$_r$($\Delta H_0$(298K),$\Delta H_1$)[kJ/mol]",fontsize=8)
plt.savefig('plot/Scatterplot_H1_H0.pdf', dpi=240)
plt.show() 

# %%
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
s = sns.scatterplot(x='H_Xe_2080', y='Enthalpy_grid_kjmol_xenon', data=df_data, s=10, alpha=0.7, ax=ax, label="Whole set (%d structures)"%len(df_data))
ax.set(ylabel="Xenon adsorption enthalpy at infinite dilution" + r"and 298K $\Delta H_0$ [kJ/mol]", 
xlabel=r"Xenon adsorption enthalpy at 1 bar and 298K $\Delta H_1$ [kJ/mol]")
ax.set_aspect('equal', adjustable='box')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color = "gray")
lim_min = -53
lim_max = -6
plt.xlim(left=lim_min,right=lim_max)
plt.ylim(bottom=lim_min,top=lim_max)
plt.show() 

# %%
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
s = sns.scatterplot(x='true', y='pred', data=df_train, s=10, alpha=0.7, ax=ax, label="Training set (%d structures)"%len(df_train))
s = sns.scatterplot(x='true', y='pred', data=df_test, s=10, alpha=0.8, ax=ax, label="Test set (%d structures)"%len(df_test))
ax.set(xlabel=r"True Free Enthalpy of exchange (1 bar) G$_1$ [kJ/mol]",
ylabel=r"ML Predicted Free Enthalpy of exchange G$_1$ [kJ/mol]")
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
plt.rcParams.update({'font.size': 10})
data = df_data[["G_2080","G_900K"]]
data['G_rd'] =  abs(data['G_2080'] - data['G_900K'])

x = data["G_2080"]
y = data["G_900K"]
z = data['G_rd']

cmap = sns.color_palette("flare", as_cmap=True)
f, ax = plt.subplots()
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", alpha=0.1, color = "gray")
points = ax.scatter(x, y, c=z, s=2, alpha=0.8, cmap=cmap)
lim_min = -15
lim_max = 30
ax.set_xlim(left=lim_min,right=lim_max)
ax.set_ylim(bottom=lim_min,top=lim_max)
plt.xlabel(r"Free Enthalpy of exchange at 1 bar G$_1$ [kJ/mol]")
plt.ylabel("Free Enthalpy of exchange\n" + r"at infinite dilution G$_0$ (900K) [kJ/mol]")
ax.set_aspect('equal', adjustable='box')
clb = f.colorbar(points)
clb.ax.set_title(r"d$_r$($\Delta H_0$(900K),$\Delta H_1$)[kJ/mol]",fontsize=8)
plt.savefig('plot/Scatterplot_G1_G900K.pdf', dpi=240)
plt.show() 

# %%
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
s = sns.scatterplot(y='G_900K', x='G_2080', data=df_data, s=10, alpha=0.7, ax=ax, label="Whole set (%d structures)"%len(df_data))
ax.set(ylabel=r"Free Enthalpy of exchange at infinite dilution G$_0$ (900K) [kJ/mol]", xlabel=r"Free Enthalpy of exchange at 1 bar G$_1$ [kJ/mol]")
# ax.set_aspect('equal', adjustable='box')
# ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color = "gray")
# lim_min = -14
# lim_max = 8
# plt.xlim(left=lim_min,right=lim_max)
# plt.ylim(bottom=lim_min,top=lim_max)
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
plt.ylabel(r"Free Enthalpy of exchange at infinite dilution G$_0$ [kJ/mol]")
plt.xlabel(r"Free Enthalpy of exchange at 1 bar G$_1$ [kJ/mol]")
ax.set_aspect('equal', adjustable='box')
clb = f.colorbar(points)
clb.ax.set_title(r"log10(Delta_VF)",fontsize=8)
# plt.savefig('plot/D_log-diameter_colored_s.pdf', dpi=240)
plt.show() 

# %%
# TODO
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


# VIOLIN PLOT // Delta
# %%
# EXPLAINABILITY
explainerModel = shap.TreeExplainer(xgb_reg)

X_shap = df_data[X_columns]
# X_shap = pd.concat([X_test, y_test], axis=1)
# X_shap = X_shap[abs(X_shap['G_0']-X_shap['G_2080'])>2][X_columns]
# X_shap = df_data[abs(df_data['G_0']-df_data['G_2080'])>2][X_columns]

top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
shap_values = explainerModel.shap_values(X_shap.to_numpy())
feature_names = X_shap.columns

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
data_test.sort_values(by="DR", ascending=False, inplace=True)

data_test[(abs(data_test["AE"])<0.5)&(abs(data_test["DR"])>1)&(data_test["true"]<-6)].head(20)

# %% selectivity drop
# n_test = 5471
n_test = 10103
# n_test = 2781
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
n_test = 643

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
xgbr = xgb.XGBRegressor(seed = random_state)

params_search = { 
           'n_estimators': [1500], 
           'max_depth': [5,6,7,8],
           'learning_rate': [0.02,0.04,0.06],
           'colsample_bytree': np.arange(0.6, 1.0, 0.05),
        #    'colsample_bytree': [0.8],
           'colsample_bylevel': np.arange(0.6, 1.0, 0.05),
        #    'colsample_bylevel': [0.6],
           'alpha': np.arange(0, 4, 0.2),
           'lambda': [0,0.5,1],
           'subsample': np.arange(0.6, 0.95, 0.05),
        #    'subsample': [0.6,0.7,0.8],
         }

s = 1
for key in params_search.keys():
    s *= len(params_search[key])
print(s)

# %%
from sklearn.model_selection import RandomizedSearchCV

gss = GroupShuffleSplit(n_splits=5, test_size=test_size,random_state=123)

clf = RandomizedSearchCV(estimator=xgbr,
                         param_distributions=params_search,
                         scoring='neg_mean_squared_error',
                         n_iter=30000,
                         verbose=1,
                         cv=gss, n_jobs=-1)
# clf = GridSearchCV(estimator=xgbr, 
#                    param_grid=params,
#                    scoring='neg_mean_squared_error', 
#                    verbose=1)

clf.fit(X_train, y_train.to_numpy(), groups=train_data['unique_chemcomp'])

# %%
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", np.sqrt(-clf.best_score_))

Best_parameters = clf.best_params_

import sys

print(clf.best_params_)

original_stdout = sys.stdout # Save a reference to the original standard output

with open('filename.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(clf.best_params_)
    sys.stdout = original_stdout # Reset the standard output to its original value
# %%
