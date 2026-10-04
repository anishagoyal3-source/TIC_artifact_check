#!/usr/bin/env python3
"""
tic_artifact_check.py- flags likely spurious TIC neighbors before a TRICERATOPS ru

For a target TIC ID it:
  1. pulls every TIC source within the search radius (MAST)
  2. pulls every Gaia DR3 source in the same patch of sky
  3. propagates TIC positions (epoch 2000) to Gaia's epoch (2016.0) and cross-matches
  4. flags neighbors that carry a TIC ARTIFACT/DUPLICATE/SPLIT disposition and/or
     have no independent Gaia DR3 counterpart (no match, or their match is
     already claimed by another TIC entry -> one of the two is a copy)

Verdicts:
  PHANTOM          bad TIC disposition AND no Gaia DR3 source      -> drop (star doesn't exist)
  DUPLICATE_COPY   bad TIC disposition AND copy of another entry   -> drop (star counted twice)
  CHECK            one warning sign only                           -> inspect by hand
  OK               none
  TARGET           the target is never dropped- its copies are

Usage:
  python tic_artifact_check.py 68035559
  python tic_artifact_check.py --file tics.txt --outdir results
  python tic_artifact_check.py --toi-survey 50 --outdir survey50
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

BAD_DISPOSITIONS = {"ARTIFACT", "DUPLICATE", "SPLIT"}
TIC_EPOCH, GAIA_EPOCH = 2000.0, 2016.0
DEFAULT_RADIUS_ARCSEC = 210.0  # 10 TESS pixels x 21"/px


# ---------------------------------------------------------------- fetching
def fetch_tic_field(tic_id, radius_arcsec):
    from astroquery.mast import Catalogs
    t = Catalogs.query_object(f"TIC {tic_id}", radius=radius_arcsec / 3600.0, catalog="TIC")
    df = t.to_pandas()
    keep = ["ID", "ra", "dec", "pmRA", "pmDEC", "Tmag", "disposition",
            "duplicate_id", "GAIA", "objType", "dstArcSec"]
    return df[[c for c in keep if c in df.columns]].copy()


def _gaia_from_vizier(ra, dec, radius_arcsec):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Gmag", "RUWE", "Dup"],
               row_limit=-1)
    res = v.query_region(SkyCoord(ra * u.deg, dec * u.deg),
                         radius=radius_arcsec * u.arcsec, catalog="I/355/gaiadr3")
    if len(res) == 0:
        return pd.DataFrame(columns=["source_id", "ra", "dec", "phot_g_mean_mag", "ruwe", "duplicated_source"])
    df = res[0].to_pandas()
    # VizieR RA_ICRS/DE_ICRS are at Gaia's epoch 2016.0
    return df.rename(columns={"Source": "source_id", "RA_ICRS": "ra", "DE_ICRS": "dec",
                              "pmRA": "pmra", "pmDE": "pmdec", "Gmag": "phot_g_mean_mag",
                              "RUWE": "ruwe", "Dup": "duplicated_source"})


def _gaia_from_esa(ra, dec, radius_arcsec):
    from astroquery.gaia import Gaia
    q = f"""
    SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, ruwe, duplicated_source
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                       CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec / 3600.0}))
    """
    df = Gaia.launch_job_async(q).get_results().to_pandas()
    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_gaia_field(ra, dec, radius_arcsec, retries=2):
    import time
    last = None
    for name, fn in [("VizieR", _gaia_from_vizier), ("ESA archive", _gaia_from_esa)]:
        for attempt in range(retries):
            try:
                df = fn(ra, dec, radius_arcsec)
                print(f"  Gaia DR3 via {name}: {len(df)} sources")
                return df
            except Exception as e:
                last = e
                print(f"  Gaia via {name} failed (try {attempt + 1}): {e}")
                time.sleep(3)
    raise RuntimeError(f"all Gaia sources failed: {last}")


EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"


def load_toi_survey(n, seed=42, cache="toi_table.csv", disposition="PC"):
    """Download (or reuse cached) ExoFOP TOI table, keep TFOPWG=PC, sample n unique TICs."""
    if not os.path.exists(cache):
        import urllib.request
        print("Downloading ExoFOP TOI table...")
        req = urllib.request.Request(EXOFOP_TOI_URL, headers={"User-Agent": "tic-artifact-check"})
        with urllib.request.urlopen(req, timeout=120) as r, open(cache, "wb") as f:
            f.write(r.read())
    toi = pd.read_csv(cache, comment="#", low_memory=False)
    col = lambda key: next(c for c in toi.columns if key in c.lower())
    tic_col, toi_col, disp_col = col("tic id"), col("toi"), col("tfopwg")
    pcs = toi[toi[disp_col].astype(str).str.strip().str.upper() == disposition]
    pcs = pcs.drop_duplicates(subset=tic_col)          # multi-planet systems: one row per star
    print(f"{len(pcs)} unique TICs with TFOPWG disposition {disposition}; sampling {min(n, len(pcs))}")
    pick = pcs.sample(n=min(n, len(pcs)), random_state=seed)
    return [str(int(t)) for t in pick[tic_col]], dict(zip(pick[tic_col].astype(int).astype(str), pick[toi_col]))


# ---------------------------------------------------------------- core logic
def propagate_tic(tic):
    """Move TIC positions from epoch 2000 to Gaia's 2016.0 using TIC proper motions."""
    dt = GAIA_EPOCH - TIC_EPOCH
    pmra = np.nan_to_num(tic.get("pmRA", 0).astype(float))    # mas/yr, includes cos(dec)
    pmdec = np.nan_to_num(tic.get("pmDEC", 0).astype(float))
    dec = tic["dec"].astype(float).values
    ra16 = tic["ra"].astype(float).values + pmra * dt / 3.6e6 / np.cos(np.radians(dec))
    dec16 = dec + pmdec * dt / 3.6e6
    return ra16, dec16


def _clean_id(x):
    """Normalize TIC IDs that may arrive as float/str/None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if s.lower() in ("", "nan", "none", "--"):
        return ""
    return s[:-2] if s.endswith(".0") else s


def audit_field(tic, gaia, target_id, match_radius_arcsec=2.0):
    """Pure function: TIC + Gaia tables in, annotated TIC table out.

    Verdicts (non-target stars):
      PHANTOM         bad TIC disposition AND no Gaia DR3 source at all   -> drop
      DUPLICATE_COPY  bad TIC disposition AND is a copy of another entry  -> drop
                      (lost a shared Gaia source, or TIC duplicate_id says so)
      CHECK           exactly one warning sign
      OK              none
    The target row is always verdict TARGET and is never dropped.
    """
    tic = tic.copy().reset_index(drop=True)
    tic["ID"] = tic["ID"].astype(str)
    target_id = str(target_id)
    is_tgt = (tic["ID"] == target_id).values
    ra16, dec16 = propagate_tic(tic)
    tic_c = SkyCoord(ra16 * u.deg, dec16 * u.deg)

    if len(gaia):
        gaia_c = SkyCoord(gaia["ra"].values * u.deg, gaia["dec"].values * u.deg)
        idx, sep, _ = tic_c.match_to_catalog_sky(gaia_c)
        sep = sep.arcsec
        matched = sep <= match_radius_arcsec
    else:
        idx = np.zeros(len(tic), int)
        sep = np.full(len(tic), np.inf)
        matched = np.zeros(len(tic), bool)

    disp = tic.get("disposition", pd.Series([""] * len(tic))).fillna("").astype(str).str.upper()
    bad = disp.isin(BAD_DISPOSITIONS).values
    tic["bad_disposition"] = bad
    tic["gaia_dr3_match"] = matched
    tic["gaia_sep_arcsec"] = np.where(matched, np.round(sep, 3), np.nan)

    has_dr2 = (tic["GAIA"].notna() if "GAIA" in tic else pd.Series(False, index=tic.index)).values
    dup_of = (tic["duplicate_id"].map(_clean_id) if "duplicate_id" in tic
              else pd.Series([""] * len(tic))).values
    id_to_row = {i: k for k, i in enumerate(tic["ID"])}
    # entries that some DUPLICATE row explicitly points to = TIC's own "this is the real one"
    referenced = np.array([i in set(dup_of[(disp == "DUPLICATE").values]) for i in tic["ID"]])

    copy_of = [""] * len(tic)

    # (1) TIC's own duplicate_id pointer, when the referenced star is in this field
    for k in range(len(tic)):
        ref = dup_of[k]
        if disp.iat[k] == "DUPLICATE" and ref and ref != tic["ID"].iat[k] and ref in id_to_row:
            if is_tgt[k]:
                # the target itself is labelled a duplicate: we keep the target
                # so the entry it points to is the redundant copy
                copy_of[id_to_row[ref]] = target_id
            else:
                copy_of[k] = ref

    # (2) several TIC entries sharing one Gaia source -> all but one are copies
    for g in np.unique(idx[matched]):
        grp = np.where(matched & (idx == g))[0]
        if len(grp) < 2:
            continue
        owner = sorted(grp, key=lambda i: (not is_tgt[i], not referenced[i], bad[i],
                                           not has_dr2[i], sep[i]))[0]
        for i in grp:
            if i != owner and not copy_of[i]:
                copy_of[i] = tic["ID"].iat[owner]
    for k in np.where(is_tgt)[0]:
        copy_of[k] = ""          # the target is never anyone's copy
    tic["copy_of"] = copy_of
    is_copy = np.array([bool(c) for c in copy_of])

    def gcol(col):
        if not len(gaia):
            return np.full(len(tic), np.nan)
        return np.where(matched, gaia[col].values[idx], np.nan)

    tic["gaia_dr3_id"] = gcol("source_id")
    tic["gaia_G"] = gcol("phot_g_mean_mag")
    tic["gaia_ruwe"] = gcol("ruwe")
    tic["gaia_dup_flag"] = gcol("duplicated_source")

    no_gaia = ~matched
    # a known copy wins over phantom
    verdict = np.where(bad & is_copy, "DUPLICATE_COPY",
              np.where(bad & no_gaia, "PHANTOM",
              np.where(bad | no_gaia | is_copy, "CHECK", "OK")))
    tic["target_issue"] = np.where(is_tgt, verdict, "")
    tic["verdict"] = np.where(is_tgt, "TARGET", verdict)
    tic["is_target"] = is_tgt

    notes = []
    for _, r in tic.iterrows():
        n = []
        if r["bad_disposition"]:
            n.append(f"TIC disposition {r.get('disposition')}")
        if not r["gaia_dr3_match"]:
            n.append(f"no DR3 source within {match_radius_arcsec}\"")
        if r["copy_of"]:
            n.append(f"copy of TIC {r['copy_of']}" + (" (THE TARGET)" if r["copy_of"] == target_id else ""))
        if r["gaia_dr3_match"] and pd.notna(r.get("Tmag")) and abs(r["gaia_G"] - r["Tmag"]) > 2:
            n.append("G vs Tmag differ >2 mag (match may be wrong star)")
        if r["gaia_dr3_match"] and r["gaia_ruwe"] > 1.4:
            n.append(f"RUWE {r['gaia_ruwe']:.2f}")
        if r["gaia_dr3_match"] and r["gaia_dup_flag"] == 1:
            n.append("Gaia duplicated_source")
        if str(r.get("objType", "STAR")).upper() not in ("STAR", "NAN", ""):
            n.append(f"objType {r.get('objType')}")
        notes.append("; ".join(n))
    tic["notes"] = notes
    return tic


def summarize(target_id, audited):
    nb = audited[~audited["is_target"]]
    t = audited[audited["is_target"]]
    t_mag = float(t["Tmag"].iloc[0]) if len(t) else np.nan
    near = nb["dstArcSec"] <= 21.0 if "dstArcSec" in nb else pd.Series(False, index=nb.index)
    out = {"tic_id": target_id, "target_Tmag": t_mag, "n_neighbors": len(nb)}
    for v, tag in [("PHANTOM", "phantom"), ("DUPLICATE_COPY", "dupcopy")]:
        s = nb[nb["verdict"] == v]
        sn = nb[(nb["verdict"] == v) & near]
        out[f"n_{tag}"] = len(s)
        out[f"n_{tag}_within_1px"] = len(sn)
        out[f"closest_{tag}_arcsec"] = s["dstArcSec"].min() if len(s) else np.nan
        out[f"brightest_near_{tag}_dmag"] = (sn["Tmag"].min() - t_mag) if len(sn) else np.nan
    out["n_check"] = int((nb["verdict"] == "CHECK").sum())
    out["copy_of_target_present"] = bool((nb["copy_of"] == str(target_id)).any())
    out["target_issue"] = t["target_issue"].iloc[0] if len(t) else ""
    out["drop_ids"] = ",".join(nb.loc[nb["verdict"].isin(["PHANTOM", "DUPLICATE_COPY"]), "ID"])
    return out


def report(target_id, audited):
    flagged = audited[(audited["verdict"] != "OK") &
                      ~((audited["verdict"] == "TARGET") & (audited["target_issue"] == "OK"))]
    print(f"\n=== TIC {target_id}: {len(audited) - 1} neighbors, "
          f"{int((audited['verdict'] != 'OK').sum() - audited['is_target'].sum())} flagged ===")
    if len(flagged):
        cols = [c for c in ["ID", "dstArcSec", "Tmag", "verdict", "notes"] if c in flagged]
        print(flagged[cols].sort_values("dstArcSec" if "dstArcSec" in cols else "ID").to_string(index=False))
    drop = audited.loc[audited["verdict"].isin(["PHANTOM", "DUPLICATE_COPY"]), "ID"].tolist()
    if drop:
        print("\nTRICERATOPS: after target = tr.target(...), run")
        print(f'  target.stars = target.stars[~target.stars["ID"].astype(str).isin({drop})].reset_index(drop=True)')


# ---------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tic", nargs="*", help="TIC ID(s)")
    p.add_argument("--file", help="text file with one TIC ID per line")
    p.add_argument("--radius", type=float, default=DEFAULT_RADIUS_ARCSEC, help="search radius, arcsec")
    p.add_argument("--match-radius", type=float, default=2.0, help="TIC-Gaia match radius, arcsec")
    p.add_argument("--outdir", default="tic_check_results")
    p.add_argument("--toi-survey", type=int, metavar="N", help="sample N TFOPWG=PC TOIs from ExoFOP")
    p.add_argument("--seed", type=int, default=42, help="random seed for the survey sample")
    a = p.parse_args()

    ids, toi_names = list(a.tic), {}
    if a.toi_survey:
        s_ids, toi_names = load_toi_survey(a.toi_survey, a.seed)
        ids += s_ids
    if a.file:
        with open(a.file) as f:
            ids += [ln.strip().replace("TIC", "").strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not ids:
        p.error("give at least one TIC ID")
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    import time
    for k, tid in enumerate(ids, 1):
        try:
            label = f" (TOI {toi_names[tid]})" if tid in toi_names else ""
            print(f"\n[{k}/{len(ids)}] TIC {tid}{label}: querying MAST TIC...")
            tic = fetch_tic_field(tid, a.radius)
            print(f"  TIC: {len(tic)} sources")
            tgt = tic[tic["ID"].astype(str) == str(tid)]
            if tgt.empty:
                raise ValueError("target not in TIC query result")
            gaia = fetch_gaia_field(float(tgt["ra"].iloc[0]), float(tgt["dec"].iloc[0]), a.radius + 30)
            audited = audit_field(tic, gaia, tid, a.match_radius)
            audited.to_csv(os.path.join(a.outdir, f"TIC{tid}_field.csv"), index=False)
            report(tid, audited)
            rows.append({**summarize(tid, audited), "toi": toi_names.get(tid, "")})
        except Exception as e:
            print(f"\n!!! TIC {tid} failed: {e}", file=sys.stderr)
            rows.append({"tic_id": tid, "toi": toi_names.get(tid, ""), "error": str(e)})
        if len(ids) > 1:
            time.sleep(1)  

    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.outdir, "summary.csv"), index=False)
    if len(ids) > 1 and "n_phantom" in summ:
        ok = summ[summ["n_phantom"].notna()]
        print(f"\n=== SURVEY: {len(ok)}/{len(summ)} targets ran ===")
        print(f"  PHANTOMS (TOI-2484 type)")
        print(f"    any in field:                         {(ok['n_phantom'] > 0).sum()}")
        print(f"    within 1 TESS pixel (21\"):           {(ok['n_phantom_within_1px'] > 0).sum()}")
        print(f"    ...and within 3 mag of target:        {(ok['brightest_near_phantom_dmag'] <= 3).sum()}")
        print(f"  DUPLICATE COPIES (double-counted real stars)")
        print(f"    any in field:                         {(ok['n_dupcopy'] > 0).sum()}")
        print(f"    within 1 TESS pixel (21\"):           {(ok['n_dupcopy_within_1px'] > 0).sum()}")
        print(f"    ...and within 3 mag of target:        {(ok['brightest_near_dupcopy_dmag'] <= 3).sum()}")
        print(f"    a copy of the TARGET itself:          {ok['copy_of_target_present'].sum()}")
        print(f"  target row itself has a warning sign:   {(ok['target_issue'] != 'OK').sum()}")
    print(f"\nSummary written to {a.outdir}/summary.csv")


if __name__ == "__main__":
    main()
