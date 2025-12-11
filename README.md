# Spatial-Transcriptomics-via-Hypergraph
The source code uses Python 3.10.

It is recommended to create a venv or other isolated package installation method.
To ensure the correct Python version is used, add -3.10 when creating the venv:
```py -3.10 -m venv .venv```

Don't forget to activate it!

The requirements.txt includes all major packages needed to run the code (dependencies not included on the list should be handled by your package installer)
With venv and pip, the packages can be installed with
```pip install -r requirements.txt```

To run the code, click "Run All" on the .ipynb's with the following order:
1. Preprocess.ipynb
2. spaformer.ipynb
3. hypergraph.ipynb
4. HGNN.ipynb

The jupyter notebooks save their results in the Data folder to serve as both a checkpoint and reusable data. This means HGNN.ipynb can be run without running the other notebooks, as long as the Data folder wasn't deleted after cloning.