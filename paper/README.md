Scripts that reproduce the paper's tables and figures.
1. Put HealthcareProviders.csv and the ProviderFraud folder where prep.py / prep_ext.py expect them (edit the paths at the top).
2. python audit.py A ; python audit.py Anocharge ; python audit.py B ; python audit.py rule5 ; python audit.py lime
3. python b_single.py ; python figs_p2.py ; python build_p2.py
