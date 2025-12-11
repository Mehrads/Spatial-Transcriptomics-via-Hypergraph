# Spatial-Transcriptomics-via-Hypergraph
The source code uses Python 3.10.

It is recommended to create a venv or other isolated package installation method.
To ensure the correct Python version is used, add -3.10 when creating the venv:
```py -3.10 -m venv .venv```

Don't forget to activate the venv!

The requirements.txt includes all major packages needed to run the code (dependencies not included on the list should be handled by your package installer)
With venv and pip, the packages can be installed with
```pip install -r requirements.txt```

dgl 0.9.1 will need to be installed manually. Please download the .whl file [here](https://pypi.org/project/dgl/0.9.1/#files)
Run pip install with the directory to the .whl file entered. For example:```pip install ./dgl-0.9.1-cp310-cp310-win_amd64.whl```

Lastly, you may be asked to install Microsoft C++ Build Tools. You can download the installer [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
You only need to install the "Desktop development with C++" and you're good to go.

The jupyter notebooks save their results in the Data folder to serve as both a checkpoint and reusable data. This means HGNN.ipynb can be run without running the other notebooks, as long as the Data folder wasn't deleted after cloning.

To run the code, run the .ipynb's with the following order:
1. Preprocess.ipynb
2. spaformer.ipynb
3. hypergraph.ipynb
4. HGNN.ipynb