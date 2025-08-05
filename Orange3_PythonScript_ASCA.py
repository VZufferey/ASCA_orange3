import os
import numpy as np
import pandas as pd
from Orange.data.pandas_compat import table_from_frame
from ASCA import ASCA_RL as ASCA
asca = ASCA.ASCA()
path = os.getcwd()

###USER INPUT
folder1 = "ASCA" ##Folder name for general ASCA outputs
folder2 =  "ASCA_subtract" ###Folder name for  subtracted effect ASCA outputs
factor_names = ['Factor1', 'Factor2', 'Treatment'] ### <<<SELECTED FACTOR COLUMNS HERE
factor_subtract_id = 0      ## None or 0-Range
print("ORANGE: Factors names: ", factor_names, "\n")
var_String= 'ENTER STRING ASSOCVIATED WITH VARIABLES COLUMNS'          ### <<<ENTER STRING TO RECOGNIZE GFETURES/VARIABLES COLUMN HERE
getPlots = True           ### False/True to toggle plotting


# Data gathering and merge, folder managment
print("Working directory: "+path)
path1 = os.path.join(path, folder1)
os.makedirs(path1, exist_ok=True) 
path2 =  os.path.join(path, folder2)
os.makedirs(path2, exist_ok=True) 
data = pd.DataFrame(in_data.X, columns=[var.name for var in in_data.domain.attributes])
meta = pd.DataFrame(in_data.metas, columns=[var.name for var in in_data.domain.metas])
df = pd.concat([data, meta], axis=1)

#Factors definition (F) : names, values matrix, and dictionnaries of values for each factor columns from entered informations
# categorical object
factor_cats = {f: pd.Categorical(df[f]) for f in factor_names}
# Numerical code
F = np.column_stack([cat.codes for cat in factor_cats.values()])
# Mapping code → label pour chaque facteur (ordre garanti)
factor_label_dicts = {fname: dict(enumerate(cat.categories)) for fname, cat in factor_cats.items()}
print(factor_label_dicts)
#Factors values matrix ASCA for analysis
F=df[factor_names].values
print(F)


#quantitative variables (X) ("NORM" in header)
variable_names = [col for col in df.columns if var_String in col]
print("ORANGE: Variables names: ", variable_names, "\n")
X = df[[col for col in df.columns if var_String in col]].values

# Define interactions to test
interactions = [[1, 2], [1, 4]]

# ASCA ANALYSIS (enable or diable plots with comment)
asca.fit(X, F, interactions=interactions, factor_names=factor_names, variable_names=variable_names)  

#ASCA output to files: factors PC plots
for i in range(len(factor_names)): #exporter un tableau du plot PCA pour chaque factor
    file_path = os.path.join(path1, "ASCA_PCplot_"+str(factor_names[i])+ ".txt")
    table = asca.getData_PCplot(Factor=i) #;print(table)
    table.to_csv(file_path, sep='\t', index=False, header=True)



########################################
#If selected, Subtraction of factor effect for 2nd ASCA analysis (ASCA2)
if factor_subtract_id is not None:
    effects = asca.getEffects(factor_subtract_id) ;print("Effects:\n", effects)
    filename = "ASCA_effect_"+asca.factor_names[factor_subtract_id]+".txt"
    file_path = os.path.join(path2, filename)
    table = pd.DataFrame(effects, columns=variable_names)
    table.to_csv(file_path, sep='\t', index=False, header=True)
    X_corrected = X - effects
    factor_names_corrected = [name + " - " + asca.factor_names[factor_subtract_id] for name in factor_names]
    variable_names_corrected = [name.replace("NORM", "-" + asca.factor_names[factor_subtract_id]) for name in variable_names]

    asca.fit(X_corrected, F, interactions=interactions, factor_names=factor_names_corrected, variable_names=variable_names_corrected)

#ASCA2 output to files
for i in range(len(factor_names)): #exporter un tableau du plot PCA pour chaque factor
    filename = "ASCA_PCplot_"+str(factor_names[i])+"-"+asca.factor_names[factor_subtract_id]+".txt"
    file_path = os.path.join(path2, filename)
    table = asca.getData_PCplot(Factor=i) #;print(table)
    table.to_csv(file_path, sep='\t', index=False, header=True)

#Plotting factors and interactions
if getPlots:
    asca.plot_factors()
    #asca.plot_interactions(interactions=interactions)

#ASCA output to orange (node ou_data), only one factor of interest
#data_table = asca.getData_PCplot(Factor=4)
#out_data = table_from_frame(data_table)
