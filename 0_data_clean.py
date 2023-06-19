import pandas as pd 
import numpy as np
import chemparse
import math 
from sklearn.model_selection import GroupShuffleSplit

# Physical constants/conversion
T = 298.0
R = 8.31446261815324e-3
P_0 = 101300          # Pa
mmol2mol = 1e-3

# Zeo++ calculation results
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

# Screening data from RASPA2 calculation
df_xe = pd.read_csv('raspa/Screening_CoReMOF_Dataset.csv')

# Energy descriptor from GrAED algorithm
df_grid_xe = pd.read_csv('graed/output_grid_0.12_298K_100_Xe.csv')
df_grid_xe['Structures'] = df_grid_xe['Structure_name'].str.rsplit('_',n=1).str[0]
df_grid_xe.drop(columns=["Structure_name"],inplace=True)
df_grid_kr = pd.read_csv('graed/output_grid_0.12_298K_100_Kr.csv')
df_grid_kr['Structures'] = df_grid_kr['Structure_name'].str.rsplit('_',n=1).str[0]
df_grid_kr.drop(columns=["Structure_name"],inplace=True)
df_psd = pd.read_csv("zeo/coremof_poredist_uff298K.csv")

df_900_Xe = pd.read_csv('graed/output_grid_0.12_900K_Xe.csv')
df_900_Xe['Structures'] = df_900_Xe['Structure_name'].str.rsplit('_',n=1).str[0]
df_900_Xe['KH_900K'] = 1e3*R*T*df_900_Xe['Henry_coeff_molkgPa']
df_900_Xe['H_900K'] = df_900_Xe['Enthalpy_grid_kjmol']
df_900_Kr = pd.read_csv('graed/output_grid_0.12_900K_Kr.csv')
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

df['K_Xe'] = df['K_Xe_widom']
df['K_Kr'] = df['K_Kr_widom']
df['H_Xe_0'] = df['H_Xe_0_widom']
df['H_Kr_0'] = df['H_Kr_0_widom']
df['Delta_H_0'] = df['H_Xe_0'] - df['H_Kr_0']
df['s_2080_log'] = np.log10(df['s_2080'])
df['Delta_H_2080'] = df['H_Xe_2080'] - df['H_Kr_2080']
df['s_0'] = df['Henry_coeff_molkgPa_xenon']/df['Henry_coeff_molkgPa_krypton']
df['s_0'] = df['s_0'].replace(0,np.nan)
df['s_2080'] = df['s_2080'].replace(0,np.nan)

df.replace([np.inf,-np.inf],np.nan,inplace=True)
df = df[~(df['DISORDER']=='DISORDER')]
print(df.shape[0])

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

# Use chemical composition to group materials
df_data["chemcomp_dict"] = df_data['chem_comp'].apply(lambda x: chemparse.parse_formula(x))
df_data['atoms_count'] = df_data["chemcomp_dict"].apply(lambda x: sum(x.values()))

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

df_data.rename(columns={'Framework Mass [g/mol]':"mass_g_mol", 'Framework Density [kg/m^3]':'density_kg_m3'}, inplace=True)

# Feature construction
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

# Save all data
df_data.to_csv("data/all_columns.csv")

# Features used in the article
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
'pore_dist_mean', 
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

# Remove nan values (search for other solutions)
# Maybe more suitable to replace by fillna with median or null values
print(len(df_data))
df_data.dropna(subset=y_column,inplace=True)
print(len(df_data))
df_data.dropna(subset=X_columns,inplace=True)
print(len(df_data))

df_data[["Structures"]+y_column+X_columns].to_csv("data/all.csv",index=False)

X, y = df_data[X_columns], df_data[y_column]

# Train/Test split on unique chemical composition
random_state=123
test_size=0.2

gss = GroupShuffleSplit(n_splits=1, test_size=test_size,random_state=123)

train_idx, test_idx = next(gss.split(X=df_data[X_columns], y=df_data[y_column], groups=df_data["unique_chemcomp"]))

train_data = df_data.reset_index().iloc[train_idx]
test_data = df_data.reset_index().iloc[test_idx]

train_data[["Structures"]+y_column+X_columns].to_csv("data/train.csv",index=False)
test_data[["Structures"]+y_column+X_columns].to_csv("data/test.csv",index=False)

