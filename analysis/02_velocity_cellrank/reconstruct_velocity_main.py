#!/usr/bin/env python3
"""
Reconstruction of the main dynamical RNA-velocity run (one attempt).

The original run that produced
    velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad
was executed interactively and the code was not saved. This script re-runs scVelo
with the parameters recorded in that object, plus one hypothesis about the missing
step, and compares the result with the original object.

Known from the original object:
    17,953 cells, 2,000 genes, moments on n_neighbors = 30, n_pcs = 30,
    92 genes fitted by recover_dynamics, 66 velocity genes (fit_likelihood > 0.1).
Known from PARAMETER_SWEEP_v2.py:
    scv.pp.filter_and_normalize(min_shared_counts=20, n_top_genes=2000)
Hypothesis tested here:
    a stochastic scv.tl.velocity was run before recover_dynamics, so the dynamical
    model was fitted only on the stochastic velocity genes.

Input: _INPUT_merged_loom.h5ad written by PARAMETER_SWEEP_v2.py (same 17,953 cells).

Result on the real data (22 Oct 2026): hypothesis confirmed at the gene level,
    fitted genes 92/92 and velocity genes 66/66 identical; per-cell Spearman
    latent time 0.972, velocity length 0.915, so NOT bit-identical. The deposited
    object remains the reference for all figures.

Run:
    export DATA_ROOT=/path/to/project_root
    conda run --no-capture-output -n b2bio python reconstruct_velocity_main.py
    # optional: --write <path.h5ad> to keep the reconstructed object
"""
import argparse, os, re, sys, time
import numpy as np
import pandas as pd
import h5py
import scvelo as scv
import scanpy as sc
from scipy.stats import spearmanr

try:
    from anndata.io import read_elem
except ImportError:  # anndata < 0.11
    from anndata.experimental import read_elem

DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
P = argparse.ArgumentParser()
P.add_argument("--input", default=DATA_ROOT + "/velocity/03_outputs/objects/parameter_sweep_v2/_INPUT_merged_loom.h5ad")
P.add_argument("--main", default=DATA_ROOT + "/velocity/03_outputs/objects/merged_3samples_dynamical_velocity__cellrank_gpcca.h5ad")
P.add_argument("--n-jobs", type=int, default=8)
P.add_argument("--write", default=None)
P.add_argument("--no-stochastic", action="store_true", help="skip the hypothesis step (control run)")
P.add_argument("--min-shared-counts", type=int, default=20)
A = P.parse_args()

BC = re.compile(r"[ACGT]{16}")


def cell_key(names, samples):
    out = []
    for n, s in zip(names, samples):
        m = BC.search(str(n))
        out.append(f"{s}_{m.group(0)}" if m else f"{s}_{n}")
    return pd.Index(out)


def vlen(V):
    V = np.asarray(V, dtype=np.float64)
    return np.sqrt(np.nansum(V ** 2, axis=1))


def read_main(path):
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"]); var = read_elem(f["var"])
        V = read_elem(f["layers/velocity"])
    if hasattr(V, "toarray"):
        V = V.toarray()
    fitted = set(var.index[np.isfinite(var["fit_alpha"].values)])
    vgenes = set(var.index[var["velocity_genes"].values.astype(bool)])
    return obs, fitted, vgenes, vlen(V)


t0 = time.time()
print("=" * 70, "\nRECONSTRUCTION RUN", "(control: no stochastic step)" if A.no_stochastic else "(hypothesis: stochastic before dynamical)")
print("=" * 70)
ad = sc.read_h5ad(A.input)
print("input:", ad.shape, "| layers:", list(ad.layers.keys()))
scv.pp.filter_and_normalize(ad, min_shared_counts=A.min_shared_counts, n_top_genes=2000)
scv.pp.moments(ad, n_pcs=30, n_neighbors=30)
if not A.no_stochastic:
    scv.tl.velocity(ad, mode="stochastic")
    print("stochastic velocity genes:", int(ad.var["velocity_genes"].sum()))
scv.tl.recover_dynamics(ad, n_jobs=A.n_jobs)
scv.tl.velocity(ad, mode="dynamical")
scv.tl.velocity_graph(ad, n_jobs=A.n_jobs)
scv.tl.latent_time(ad)
fitted_new = set(ad.var_names[np.isfinite(ad.var["fit_alpha"].values)])
vg_new = set(ad.var_names[ad.var["velocity_genes"].values.astype(bool)])
vl_new = vlen(ad.layers["velocity"])
print(f"done in {(time.time() - t0) / 60:.1f} min")

print("=" * 70, "\nCOMPARISON WITH ORIGINAL OBJECT\n" + "=" * 70)
obs_m, fitted_m, vg_m, vl_m = read_main(A.main)
print(f"fitted genes    : original {len(fitted_m)}  reconstructed {len(fitted_new)}  shared {len(fitted_m & fitted_new)}")
print(f"velocity genes  : original {len(vg_m)}  reconstructed {len(vg_new)}  shared {len(vg_m & vg_new)}")

km = cell_key(obs_m.index, obs_m["sample"].astype(str))
kn = cell_key(ad.obs_names, ad.obs["sample"].astype(str))
common = km.intersection(kn)
im = pd.Series(np.arange(len(km)), index=km)[common].values
inn = pd.Series(np.arange(len(kn)), index=kn)[common].values
print(f"matched cells   : {len(common)} of {len(km)}")

lt_m = obs_m["latent_time"].values[im]; lt_n = ad.obs["latent_time"].values[inn]
s = obs_m["sample"].astype(str).values[im]
rho_lt = spearmanr(lt_m, lt_n).correlation
rho_vl = spearmanr(vl_m[im], vl_new[inn]).correlation
print(f"per-cell Spearman: latent time {rho_lt:.4f}   velocity length {rho_vl:.4f}")
print("\nmedians per sample (all matched cells)      original -> reconstructed")
for smp in sorted(set(s)):
    k = s == smp
    print(f"  {smp:5s}  latent time {np.median(lt_m[k]):.3f} -> {np.median(lt_n[k]):.3f}"
          f"   velocity length {np.median(vl_m[im][k]):.2f} -> {np.median(vl_new[inn][k]):.2f}")

exact = fitted_new == fitted_m and vg_new == vg_m and rho_lt > 0.999
print("\nVERDICT:", "EXACT (same genes, latent time rho > 0.999)" if exact
      else "NOT EXACT (see differences above)")
if A.write:
    ad.write(A.write); print("written:", A.write)
sys.exit(0 if exact else 2)
