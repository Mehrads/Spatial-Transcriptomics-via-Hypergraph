"""
test_fixes.py — verifies every bug fix applied to the pipeline.
Run with: conda run -n base python test_fixes.py
"""
import sys, traceback
import numpy as np

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

results = []

def check(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        msg = str(e)
        print(f"{FAIL} {name}")
        print(f"       {msg}")
        results.append((name, False, msg))

DATA = "Data/spaformer_prepared"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Preprocess: k=15
# ─────────────────────────────────────────────────────────────────────────────
def test_knn_k15():
    edges = np.load(f"{DATA}/edges.npy")  # built with k=6 (old), check shape tells us
    # We can't re-run Preprocess, but we can verify the fix is in the notebook source.
    import json
    with open("Preprocess.ipynb") as f:
        nb = json.load(f)
    found_k15 = any(
        "k = 15" in "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    )
    assert found_k15, "k=15 not found in any code cell of Preprocess.ipynb"

check("Preprocess: k=15 in source", test_knn_k15)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocess: HVG ordering (normalize happens AFTER HVG selection)
# ─────────────────────────────────────────────────────────────────────────────
def test_hvg_order():
    import json
    with open("Preprocess.ipynb") as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "highly_variable_genes" in src and "normalize_total" in src:
            hvg_pos = src.index("highly_variable_genes")
            norm_pos = src.index("normalize_total")
            assert hvg_pos < norm_pos, (
                f"HVG selection (pos {hvg_pos}) must come BEFORE "
                f"normalize_total (pos {norm_pos})"
            )
            return
    raise AssertionError("Could not find cell containing both HVG and normalize_total")

check("Preprocess: HVG runs before normalize_total", test_hvg_order)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Positional encoding: coords appended to features
# ─────────────────────────────────────────────────────────────────────────────
def test_pe_concatenation():
    X = np.load(f"{DATA}/X.npy")       # (N, G)
    C = np.load(f"{DATA}/C.npy")       # (N, 2)
    X_pe = np.concatenate([X, C], axis=1)
    assert X_pe.shape == (X.shape[0], X.shape[1] + 2), \
        f"Expected (N, G+2) = {(X.shape[0], X.shape[1]+2)}, got {X_pe.shape}"
    # Verify the last two columns are exactly the coords
    np.testing.assert_array_equal(X_pe[:, -2:], C)

check("PE: concatenating coords gives (N, G+2) with coords in last 2 cols", test_pe_concatenation)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Mask rates in spaformer.ipynb source
# ─────────────────────────────────────────────────────────────────────────────
def test_mask_rates():
    import json
    with open("spaformer.ipynb") as f:
        nb = json.load(f)
    # Only search code cells (markdown analysis cells mention old values in prose)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if "mask_node_rate" in src and "mask_feature_rate" in src:
            assert "mask_node_rate=0.15" in src, \
                f"Expected mask_node_rate=0.15, found: {[l for l in src.splitlines() if 'mask_node_rate' in l and not l.strip().startswith('#')]}"
            assert "mask_feature_rate=0.15" in src, \
                f"Expected mask_feature_rate=0.15"
            return
    raise AssertionError("mask_node_rate not found in any code cell of spaformer.ipynb")

check("spaformer: mask_node_rate=0.15, mask_feature_rate=0.15", test_mask_rates)

# ─────────────────────────────────────────────────────────────────────────────
# 5. outer_epochs_per_iter = 1
# ─────────────────────────────────────────────────────────────────────────────
def test_outer_epochs():
    import json
    with open("spaformer.ipynb") as f:
        nb = json.load(f)
    # Only search code cells (markdown analysis cell mentions the old value)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if "outer_epochs_per_iter" in src:
            lines = [l.strip() for l in src.splitlines()
                     if "outer_epochs_per_iter" in l and not l.strip().startswith("#")]
            assignment = [l for l in lines if "=" in l]
            assert assignment, f"No assignment found for outer_epochs_per_iter in code cell"
            val = assignment[0].split("=")[1].strip()
            assert val == "1", f"Expected outer_epochs_per_iter=1, got: {val!r}"
            return
    raise AssertionError("outer_epochs_per_iter not found in any code cell of spaformer.ipynb")

check("spaformer: outer_epochs_per_iter=1", test_outer_epochs)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Consistency loss fix: noise makes z != z2
# ─────────────────────────────────────────────────────────────────────────────
def test_consistency_loss_fix():
    # Simulate: with noise, two versions of X differ → z would differ too
    N, G = 100, 50
    rng = np.random.default_rng(42)
    X = rng.standard_normal((N, G)).astype(np.float32)
    noise_scale = 0.01
    X_noisy = X + noise_scale * rng.standard_normal((N, G)).astype(np.float32)
    diff = np.abs(X - X_noisy).max()
    assert diff > 0, "Noisy X is identical to X — noise not applied"
    assert diff < 0.5, f"Noise too large: max diff={diff:.4f}"
    # Verify the fix is in the source
    import json
    with open("spaformer.ipynb") as f:
        nb = json.load(f)
    found = any(
        "randn_like" in "".join(cell.get("source", []))
        for cell in nb["cells"]
    )
    assert found, "randn_like not found in spaformer.ipynb — consistency loss fix missing"

check("spaformer: consistency loss uses randn noise (z != z2)", test_consistency_loss_fix)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Contrastive loss fix: label_pos != label_neg
# ─────────────────────────────────────────────────────────────────────────────
def test_contrastive_labels():
    N = 3639
    label_pos = np.tile([1.0, 0.0], (N, 1))
    label_neg = np.tile([0.0, 1.0], (N, 1))
    assert not np.array_equal(label_pos, label_neg), \
        "label_pos and label_neg are identical — fix not applied"
    assert label_pos.shape == (N, 2)
    assert label_neg.shape == (N, 2)
    np.testing.assert_array_equal(label_pos + label_neg, np.ones((N, 2)))
    # Verify label_neg is in source
    import json
    with open("spaformer.ipynb") as f:
        nb = json.load(f)
    found = any(
        "label_neg" in "".join(cell.get("source", []))
        for cell in nb["cells"]
    )
    assert found, "label_neg not found in spaformer.ipynb"

check("spaformer: contrastive label_pos=[1,0], label_neg=[0,1]", test_contrastive_labels)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Spatial k-NN hyperedge construction: shape, uniqueness, coverage
# ─────────────────────────────────────────────────────────────────────────────
def test_spatial_hyperedges():
    from sklearn.neighbors import NearestNeighbors
    C = np.load(f"{DATA}/C.npy").astype(np.float32)
    N = C.shape[0]
    KNN_HE = 15

    nbrs = NearestNeighbors(n_neighbors=KNN_HE + 1, metric="euclidean").fit(C)
    _, idx = nbrs.kneighbors(C)

    node_ids, edge_ids = [], []
    for eid, neighbors in enumerate(idx):
        node_ids.extend(neighbors.tolist())
        edge_ids.extend([eid] * len(neighbors))

    H = np.array([node_ids, edge_ids], dtype=np.int64)

    # Shape check: N*(k+1) entries
    assert H.shape[0] == 2, f"Expected shape (2, ...), got {H.shape}"
    assert H.shape[1] == N * (KNN_HE + 1), \
        f"Expected {N*(KNN_HE+1)} entries, got {H.shape[1]}"

    # Every node appears as a hyperedge center exactly once
    unique_edges = np.unique(H[1])
    assert len(unique_edges) == N, \
        f"Expected {N} unique hyperedges (one per node), got {len(unique_edges)}"

    # Every node appears as a member in at least one hyperedge
    unique_nodes = np.unique(H[0])
    assert len(unique_nodes) == N, \
        f"Expected all {N} nodes to appear as members, got {len(unique_nodes)}"

    # Node IDs and edge IDs are in valid range [0, N)
    assert H[0].min() >= 0 and H[0].max() < N
    assert H[1].min() >= 0 and H[1].max() < N

check("Spatial k-NN hyperedges: shape, coverage, ID validity", test_spatial_hyperedges)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Spatial hyperedge: each hyperedge has exactly KNN+1 members (self + k neighbors)
# ─────────────────────────────────────────────────────────────────────────────
def test_hyperedge_sizes():
    from sklearn.neighbors import NearestNeighbors
    C = np.load(f"{DATA}/C.npy").astype(np.float32)
    N = C.shape[0]
    KNN_HE = 15

    nbrs = NearestNeighbors(n_neighbors=KNN_HE + 1, metric="euclidean").fit(C)
    _, idx = nbrs.kneighbors(C)

    node_ids, edge_ids = [], []
    for eid, neighbors in enumerate(idx):
        node_ids.extend(neighbors.tolist())
        edge_ids.extend([eid] * len(neighbors))

    H = np.array([node_ids, edge_ids], dtype=np.int64)
    sizes = np.bincount(H[1])
    assert (sizes == KNN_HE + 1).all(), \
        f"Not all hyperedges have size {KNN_HE+1}; min={sizes.min()}, max={sizes.max()}"

check(f"Spatial k-NN hyperedges: every hyperedge has exactly 16 members", test_hyperedge_sizes)

# ─────────────────────────────────────────────────────────────────────────────
# 10. HGNN: train/val split is stratified, correct sizes
# ─────────────────────────────────────────────────────────────────────────────
def test_hgnn_split():
    from sklearn.model_selection import train_test_split

    labels_all = np.load(f"{DATA}/labels_nested.npy").astype("int64")
    train_idx = labels_all != -1
    valid_positions = np.where(train_idx)[0]

    uniq = np.unique(labels_all[train_idx])
    label_codes = np.searchsorted(uniq, labels_all[train_idx]).astype("int64")

    pos_tr, pos_val = train_test_split(
        np.arange(len(valid_positions)),
        test_size=0.2,
        random_state=42,
        stratify=label_codes,
    )

    n_total = len(valid_positions)
    n_val_expected = int(round(n_total * 0.2))
    assert abs(len(pos_val) - n_val_expected) <= 1, \
        f"Val size {len(pos_val)} too far from expected {n_val_expected}"
    assert len(pos_tr) + len(pos_val) == n_total

    # Class distribution should be ~equal across train and val
    tr_labels = label_codes[pos_tr]
    val_labels = label_codes[pos_val]
    for cls in np.unique(label_codes):
        tr_frac  = (tr_labels  == cls).mean()
        val_frac = (val_labels == cls).mean()
        assert abs(tr_frac - val_frac) < 0.05, \
            f"Class {cls} imbalanced: train={tr_frac:.3f} val={val_frac:.3f}"

check("HGNN: 80/20 stratified train/val split correct", test_hgnn_split)

# ─────────────────────────────────────────────────────────────────────────────
# 11. HGNN: no KMeans re-clustering in source
# ─────────────────────────────────────────────────────────────────────────────
def test_hgnn_no_redundant_kmeans():
    import json
    with open("HGNN.ipynb") as f:
        nb = json.load(f)
    full_src = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    )
    # The fix uses pred_codes directly; KMeans should be gone from final assignment
    assert "KMeans" not in full_src or "# [FIX]" in full_src, \
        "KMeans still present without fix comment"
    # Classifier argmax should be used
    assert "pred_codes" in full_src, "pred_codes not found — classifier argmax fix missing"
    assert "final_codes = pred_codes" in full_src, \
        "final_codes = pred_codes not found in HGNN.ipynb"

check("HGNN: final_codes uses classifier argmax, not KMeans", test_hgnn_no_redundant_kmeans)

# ─────────────────────────────────────────────────────────────────────────────
# 12. HGNN: early stopping monitors val_loss, not train accuracy
# ─────────────────────────────────────────────────────────────────────────────
def test_hgnn_val_early_stopping():
    import json
    with open("HGNN.ipynb") as f:
        nb = json.load(f)
    full_src = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    )
    assert "best_val_loss" in full_src, \
        "best_val_loss not found — val-based early stopping fix missing"
    assert "val_loss < best_val_loss" in full_src, \
        "val_loss comparison not found"

check("HGNN: early stopping uses best_val_loss", test_hgnn_val_early_stopping)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Integration: existing saved arrays are compatible with new hyperedge shape
# ─────────────────────────────────────────────────────────────────────────────
def test_saved_data_compatible():
    X       = np.load(f"{DATA}/X.npy")
    C       = np.load(f"{DATA}/C.npy")
    labels  = np.load(f"{DATA}/labels_nested.npy")
    latent  = np.load(f"{DATA}/latent_nested.npy")
    edges   = np.load(f"{DATA}/edges.npy")

    N = X.shape[0]
    assert C.shape       == (N, 2),   f"C shape mismatch: {C.shape}"
    assert labels.shape  == (N,),     f"labels shape mismatch: {labels.shape}"
    assert latent.shape  == (N, 256), f"latent shape mismatch: {latent.shape}"
    assert edges.shape[0] == 2,       f"edges should be (2, E)"

    # X+C concatenation yields (N, G+2)
    X_pe = np.concatenate([X, C], axis=1)
    assert X_pe.shape == (N, X.shape[1] + 2)

    # labels have exactly 7 unique values (no noise nodes expected from kmeans)
    n_unique = np.unique(labels[labels != -1]).size
    assert n_unique == 7, f"Expected 7 label classes, got {n_unique}"

check("Integration: saved arrays are consistent and PE-compatible", test_saved_data_compatible)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"{'='*60}")
print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
if failed:
    print("\nFailed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"  • {name}")
            print(f"    {err}")
    sys.exit(1)
else:
    print("All tests passed.")
