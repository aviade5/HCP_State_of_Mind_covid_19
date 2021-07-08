# HCP_State_of_Mind_covid_19

- The folder **classifiers** contains pickle files representing the Account Type and Occupation Type classfiers, which can be loaded as scikit-learn SVM classifiers.
- The folder **code** contains three folders:
	- **manual_labeling** - code related exclusively to the process of creating the classifiers, manually labeling accounts and the evaluation process in general.
	-  **hpc_code** - code that ran on HPC clusters due to resource limitations when using Jupyter Notebooks. 
	- **topic_modeling_notebooks** - contains all Jupyter Notebooks used for analysis and figure creation. Some of the Notebooks were migrated to .py files found in hpc_code due to insufficient resources to run the Notebooks on non-super-computing machines.
-   The folder **graphics** contains all figures shown in the paper and in the appendix.