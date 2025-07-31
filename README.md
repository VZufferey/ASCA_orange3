# ASCA_orange3
ANOVA–simultaneous component analysis (ASCA) python script for orange3 Data Mining 

Files:
Orange 3 pipeline .ows
Pthon script node(included in pipeline): ASCA_Orange3.py
ASCA analysis from customized ASCA library: needs separate installation of package. 
	a. install vanilla library and add/replace ASCA.py so that ASCA_RL.py is used.


ASCA Analysis in Orange3:
1. Copy the orange pipeline at the location of your data analysis.

2. Import and review your data (possibility to merge 2 tables for a labelling process, or to directly  ready to use dataset). Defin your grouping columns as categorical.

3. Filter the data if needed (subset analysis, ...) 

4. Make sure the factor are formatted as text (edit domain)

5. Adapt the python script (in the python script node): Enter the factor column names (factor_names) (coresponds togrouping columns, qualitative data..), and the string (var_String) that must be used to detect the feature columns (coresponds to the cquantitative values)

6. plots should appear, and data tables should be saved at the location of the orange pipeline, in the /ASCA Folder 

Source script: 
https://pypi.org/project/ASCA/

Smilde, Age K., et al. "ANOVA-simultaneous component analysis (ASCA): a new tool for analyzing designed metabolomics data." Bioinformatics 21.13 (2005): 3043-3048.

Jansen, Jeroen J., et al. "ASCA: analysis of multivariate data obtained from an experimental design." Journal of Chemometrics: A Journal of the Chemometrics Society 19.9 (2005): 469-481.
