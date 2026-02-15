# Spatial-Transcriptomics-via-Hypergraph
The source code targets Python 3.10.

It is recommended to create a virtual environment:
`py -3.10 -m venv .venv`

Install project dependencies:
`pip install -r requirements.txt`

Leiden clustering in Scanpy requires `igraph` + `leidenalg` (included in `requirements.txt`).

SpaFormer now uses PyTorch Geometric (PyG) by default in `spaformer.ipynb`.
DGL is optional and only needed if you explicitly set `USE_PYG=False` in the notebook.
For multi-GPU training in SpaFormer, keep `USE_PYG=True`, `device='auto'` (or `'cuda'`), and `use_all_gpus=True`.

## PyG install notes (CPU/CUDA)
1. Install a matching PyTorch build first (CPU or CUDA) from https://pytorch.org/get-started/locally/.
2. Install PyG packages that match your Torch/CUDA build using https://data.pyg.org/whl/.
3. CPU example:
`pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html`
4. CUDA example:
`pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html`
5. Verify:
`python -c "import torch, torch_geometric; print(torch.__version__, torch.cuda.is_available())"`

Notebook outputs are saved under `Data/` as reusable artifacts/checkpoints.

Run notebooks in this order:
1. `Preprocess.ipynb`
2. `spaformer.ipynb`
3. `hypergraph.ipynb`
4. `HGNN.ipynb`
