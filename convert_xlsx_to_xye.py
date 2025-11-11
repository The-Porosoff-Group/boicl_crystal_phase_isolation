
import math
from pathlib import Path
import pandas as pd

DATA_DIR = Path("/Users/shane/Downloads/XRD Index")

def choose_intensity_column(df):
    """
    Prefer 'intx' if it has any nonzero values; otherwise 'count';
    otherwise first numeric column (excluding 2thetadeg/d-value) that has nonzero values.
    """
    for cand in ["intx", "count"]:
        if cand in df.columns:
            col = pd.to_numeric(df[cand], errors="coerce").fillna(0)
            if col.ne(0).any():
                return cand

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric:
        if c.lower() in {"2thetadeg", "d-value"}:
            continue
        col = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if col.ne(0).any():
            return c
    return None

def convert_one_excel(xls_path: Path) -> dict:
    
    df = pd.read_excel(xls_path, header=0)

    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    if "2thetadeg" not in cols:
        return {"file": xls_path.name, "status": "skip", "reason": "no '2thetadeg' column"}

    inten_col = choose_intensity_column(df.rename(columns={v: k for k, v in cols.items()}))
    if inten_col is None:
        return {"file": xls_path.name, "status": "skip", "reason": "no usable intensity column"}

    inten_col = cols.get(inten_col, inten_col)

    tth   = pd.to_numeric(df[cols["2thetadeg"]], errors="coerce")
    inten = pd.to_numeric(df[inten_col],         errors="coerce")

    use_sigx = "sigx" in cols
    if use_sigx:
        sig = pd.to_numeric(df[cols["sigx"]], errors="coerce").fillna(0)
        use_sigx = (sig > 0).any()
    sigma = sig.where(sig > 0, inten.clip(lower=1.0).pow(0.5)) if use_sigx else inten.clip(lower=1.0).pow(0.5)

    valid = tth.notna() & inten.notna() & sigma.notna()
    out = pd.DataFrame({"2theta": tth[valid], "I": inten[valid], "sigma": sigma[valid]})
    out = out.sort_values("2theta").reset_index(drop=True)

    if out.empty or (out["I"].fillna(0) == 0).all():
        return {"file": xls_path.name, "status": "skip", "reason": "no nonzero intensities"}

    out_path = xls_path.with_suffix(".xye")
    with open(out_path, "w") as f:
        for a, b, e in out.itertuples(index=False):
            f.write(f"{a:.6f} {b:.6f} {e:.6f}\n")

    return {
        "file": xls_path.name,
        "status": "ok",
        "out": out_path.name,
        "points": len(out),
        "tth_min": float(out["2theta"].min()),
        "tth_max": float(out["2theta"].max()),
        "intensity_col": inten_col,
        "sigma_used": "sigx" if use_sigx else "sqrt(I)",
    }


results = []
for fn in sorted(DATA_DIR.iterdir()):
    if fn.suffix.lower() not in (".xlsx", ".xls"):
        continue
    try:
        res = convert_one_excel(fn)
    except Exception as e:
        res = {"file": fn.name, "status": "error", "reason": str(e)}
    print(res)
    results.append(res)

# optional
if results:
    pd.DataFrame(results).to_csv(DATA_DIR / "conversion_summary.csv", index=False)
    print(f"\nSummary written to: {DATA_DIR / 'conversion_summary.csv'}")
else:
    print("No Excel files found to convert.")
