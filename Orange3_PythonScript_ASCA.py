import os
import numpy as np
import pandas as pd
from Orange.data.pandas_compat import table_from_frame
from ASCA import ASCA

#meta
asca = ASCA.ASCA()
path=os.path.join(os.getcwd(), "ASCA")
print("Working directory: "+path)

# Data gathering and merge
data = pd.DataFrame(in_data.X, columns=[var.name for var in in_data.domain.attributes])
meta = pd.DataFrame(in_data.metas, columns=[var.name for var in in_data.domain.metas])
df = pd.concat([data, meta], axis=1)

factor_names = ['Factor 1', 'Factor2', '...'] ### <<<SELECTED FACTOR COLUMNS HERE
print("ORANGE: Factors names: ", factor_names, "\n")
var_String= 'factor_' ### <<<ENTER STRING TO RECOGNIZE GFETURES/VARIABLES COLUMN HERE

#Factors definition (F) : names, values matrix, and dictionnaries of values for each factor columns from entered informations
# categorical object
factor_cats = {f: pd.Categorical(df[f]) for f in factor_names}
# Numerical code
F = np.column_stack([cat.codes for cat in factor_cats.values()])
# Mapping code → label pour chaque facteur (ordre garanti)
factor_label_dicts = {fname: dict(enumerate(cat.categories)) for fname, cat in factor_cats.items()}
print(factor_label_dicts)
#Factors values ASCA for analysis
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

#Plotting factors and interactions
#asca.plot_factors()
#asca.plot_interactions(interactions=interactions)

#ASCA output to files: PC summary matrices (residuals, ...)
residuals = asca.getResiduals()
file_path = os.path.join(path, "ASCA_residuals.txt")
table = pd.DataFrame(residuals, columns=variable_names)
print("residuals: ", table)
table.to_csv(file_path, sep='\t', index=False, header=True)

#ASCA output to files: factors PC plots
for i in range(len(factor_names)): #exporter un tableau du plot PCA pour chaque factor
    table = asca.getData_PCplot(Factor=i) #;print(table)
    file_path = os.path.join(path, "ASCA_PCplot_"+str(factor_names[i])+ ".txt")
    table.to_csv(file_path, sep='\t', index=False, header=True)

#ASCA output to orange (node ou_data), only one factor of interest
#data_table = asca.getData_PCplot(Factor=4)
#out_data = table_from_frame(data_table)

