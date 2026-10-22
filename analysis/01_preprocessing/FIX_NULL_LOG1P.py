"""
Popravak: brise 'uns/log1p/base' (null-encoded) iz starih h5ad fajlova
da bi anndata 0.11.4 mogla da ih ucita.

Pravi BACKUP svakog fajla pre izmene.
"""
import os  # added by sanitize_paths.py
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")  # project root, see README
import os
import shutil
import h5py
import scanpy as sc

PATHS = [
    DATA_ROOT + "/scanpy_out_MDA231/objects/MDA231_stage1_adata.h5ad",
    DATA_ROOT + "/scanpy_out_MDA231/objects/MDA231_stage1_adata_WITHRAW.h5ad",
    DATA_ROOT + "/scanpy_out_MDA231/objects/MDA231_stage6b_trajectory.h5ad",
]

for path in PATHS:
    print("=" * 70)
    print(f"PROCESSING: {path}")

    if not os.path.exists(path):
        print("  -> NE POSTOJI, preskace")
        continue

    backup = path + ".bak_before_null_fix"
    if not os.path.exists(backup):
        print(f"  -> backup: {backup}")
        shutil.copy2(path, backup)
    else:
        print(f"  -> backup vec postoji: {backup}")

    # --- Step 1: pronadji null-encoded zapise (h5py)
    null_paths = []
    with h5py.File(path, "r") as f:
        def find_null(name, obj):
            if dict(obj.attrs).get("encoding-type") == "null":
                null_paths.append(name)
        f.visititems(find_null)

    print(f"  -> nadjeno null objekata: {len(null_paths)}")
    for p in null_paths:
        print(f"     - {p}")

    if not null_paths:
        print("  -> nista za popraviti, preskace")
        continue

    # --- Step 2: izbrisi ih (otvaranje u 'a' modu = read+write)
    with h5py.File(path, "a") as f:
        for p in null_paths:
            try:
                del f[p]
                print(f"  -> obrisano: {p}")
            except Exception as e:
                print(f"  -> GRESKA brisanja {p}: {e}")

    # --- Step 3: probaj da ucitas u scanpy da potvrdis
    try:
        ad = sc.read_h5ad(path)
        print(f"  -> SADA SE UCITAVA: shape={ad.shape}")
        print(f"     obs columns: {list(ad.obs.columns)[:6]}...")
        print(f"     X min/max: {float(ad.X.min()):.3f} / {float(ad.X.max()):.3f}")
        if ad.raw is not None:
            print(f"     raw shape: {ad.raw.X.shape}")
        else:
            print(f"     raw: None")
    except Exception as e:
        print(f"  -> JOS UVEK PUCA: {e}")
        print(f"  -> backup je u {backup}, mozes vratiti sa: cp '{backup}' '{path}'")

print("\nGotovo.")
