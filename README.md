**tic_artifact_check**

A pre-flight check for TRICERATOPS users. It cross-matches the TESS Input Catalog (TIC) around a target against Gaia DR3 to flag **phantom stars** and **duplicate catalog entries** before they distort false-positive probability (FPP) calculations.

**Why**

TRICERATOPS treats every TIC star near a target as a possible source of the transit. But the TIC contains entries that don't correspond to real, separate stars. While validating TOI-2484, a phantom TIC neighbor (TIC 68035558, 4.17" from the target, Tmag 11.5, no Gaia counterpart) inflated the FPP from 0.27 to 0.48. This tool catches that kind of error automatically.

**How it works**

For each target, the tool:

1. Queries every TIC source within 210" (TRICERATOPS's default field) from MAST.
2. Queries Gaia DR3 for the same patch of sky (via VizieR, with the ESA archive as a fallback).
3. Propagates TIC positions from epoch 2000 to Gaia's epoch 2016.0 using proper motions.
4. Cross-matches each TIC star to its nearest Gaia source within 2".
5. Assigns a verdict to each neighbor:

| Verdict | Meaning | Action |
|---|---|---|
| Phantom | Bad TIC disposition (ARTIFACT/DUPLICATE/SPLIT) **and** no Gaia source | Drop: the star doesn't exist |
| Duplicate Copy | Bad TIC disposition **and** a copy of another entry (shares its Gaia source, or TIC's Dupicate ID says so) | Drop: the star is counted twice |
| Check | Exactly one warning sign | Inspect by hand |
| ok | No warning signs | Keep |
| Target | The target itself | Never dropped; copies of it are |

A star is only dropped when two independent lines of evidence agree, because wrongly removing a real star would make a false positive look more planet-like.

**Installation**

```
pip install numpy pandas astropy astroquery
```

**Usage**

**Check one target** (the main use):
```
python tic_artifact_check.py 68035559
```
This prints the flagged stars and the exact line to paste into your TRICERATOPS script to remove them.

**Check a list of targets** (one TIC ID per line):
```
python tic_artifact_check.py --file my_targets.txt
```

**Survey random planet candidates** from the ExoFOP TOI table:
```
python tic_artifact_check.py --toi-survey 300 --outdir survey300
```

Options: `--radius` (search radius, arcsec), `--match-radius` (TIC–Gaia match radius, arcsec), `--seed` (survey sample seed), `--outdir`.

Outputs: one `TIC<id>_field.csv` per target and a `summary.csv` across all targets.

**Survey results**

A random sample of 294 TESS planet candidates (TFOPWG disposition PC, seed 42) with 95% Wilson intervals:

| Catalog problem | Rate |
|---|---|
| Phantom within 1 TESS pixel (21") | 1.4% (0.5–3.4%) |
| Bright phantom within 1 pixel and 3 mag of target | 0/294 (< 1.3%) |
| Duplicate copy within 1 pixel and 3 mag of target | 3.7% (2.1–6.6%) |
| **Duplicate copy of the target star itself** | **4.1% (2.4–7.0%)** |

For TOI-7136.01, TRICERATOPS loads the duplicate copy of the target into its stellar field at 0" separation with identical brightness, and lists it ahead of the real target.

**Not yet tested:** how much these entries change TRICERATOPS FPP values in practice.

**Limitations**

- Gaia misses some real stars (very faint, very bright, or tight pairs under ~1"), which is why a missing Gaia match alone only produces `CHECK`.
- TIC dispositions were assigned automatically and can be wrong.
- Results depend on MAST and VizieR availability; failed queries are logged in `summary.csv` and can be rerun with `--file`.

## Author

Anisha Goyal
