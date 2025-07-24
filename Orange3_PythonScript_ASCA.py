import numpy as np
from ASCA import ASCA
import numpy as np
import pandas as pd

print("\nASCA import: ", ASCA, "\n")
asca = ASCA.ASCA()

# Data gathering
data = pd.DataFrame(in_data.X, columns=[var.name for var in in_data.domain.attributes])
meta = pd.DataFrame(in_data.metas, columns=[var.name for var in in_data.domain.metas])

# Data merge
df = pd.concat([data, meta], axis=1)

#Selection of quantitative variables (X)("NORM" Dans header) and factors  (F) (variables de groupes)
X = df[[col for col in df.columns if 'NORM' in col]].values;
print("X=", X)

F=df[['Replicate', 'Culture', 'mouse ID', 'TimeLapseIndex', 'Treatment', 'Condition unique','Well', 'Condition mouse']].values
print("F=", F)

# Define interactions to test
interactions = [[1, 8]]

# ASCA analysis
asca.fit(X, F, interactions)

# Principal effects visualisation
asca.plot_factors()
asca.plot_interactions()
