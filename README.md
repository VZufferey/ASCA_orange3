# ASCA_orange3
ANOVA–simultaneous component analysis (ASCA) python script for orange3 Data Mining 

Files:
Orange 3 pipeline .ows
Pthon script node(included in pipeline): ASCA_Orange3.py
ASCA analysis from customized ASCA library: needs separate installation of package. 
	a. install vanilla library and add/replace ASCA.py so that ASCA_RL.py is used.

*Installation for Windows:*

ASCA can be installed with >>>pip install ASCA (more info on ASCA page, source below), but where you enter this command will depend on how Orange3 is installed on your machine.

Vanilla ASCA library installation for orange3's python scripts --> package localization dependending on python setup	
- If you have installed orange3 with the installer (default orange environment), use the shortcut to the Python console in the Orange installation folder, 
- If you installed orange3 with python commands, in a custom python environment (Conda, , ...), activate your environment, and run the command	
  
locate ...\Lib\site-packages\ASCA at the location of the environment, and copy ASCA_RL.py here (this step alone can done to install ASCA_RL only)
   

*ASCA Analysis in an Orange3 pipeline:*
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
