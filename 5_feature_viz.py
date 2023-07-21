# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

random_state = 123
T = 298.0
R = 8.31446261815324e-3

# %%
# import training data
df_data = pd.read_csv('data/all_columns.csv',index_col=0)
X_data, y_data, group = df_data.iloc[:,3:], df_data['G_2080'], df_data["unique_chemcomp"]

df_test = pd.read_csv('data/test.csv',index_col=0)
X_test, y_test, group_test = df_test.iloc[:,3:], df_test['G_2080'], df_test["unique_chemcomp"]
feature_names = list(X_test.columns)

# %%
plt.figure(figsize=(18,18))
cor = df_data[["G_2080"]+feature_names].corr()
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
plt.xlabel(r"Gibbs free energy of exchange at 1 bar G$_1$ [kJ/mol]")
plt.ylabel("Gibbs free energy of exchange\n" + r"at infinite dilution G$_0$ (900K) [kJ/mol]")
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
ax.set(ylabel=r"Gibbs free energy of exchange at infinite dilution G$_0$ (900K) [kJ/mol]", xlabel=r"Gibbs free energy of exchange at 1 bar G$_1$ [kJ/mol]")
# ax.set_aspect('equal', adjustable='box')
# ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color = "gray")
# lim_min = -14
# lim_max = 8
# plt.xlim(left=lim_min,right=lim_max)
# plt.ylim(bottom=lim_min,top=lim_max)
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
plt.ylabel(r"Gibbs free energy of exchange at infinite dilution G$_0$ [kJ/mol]")
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
