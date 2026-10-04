import numpy as np
import pandas as pd
from tic_artifact_check import audit_field, summarize

RA0, DEC0 = 250.0, 30.0


def offset(dra, ddec):
    return RA0 + dra / 3600 / np.cos(np.radians(DEC0)), DEC0 + ddec / 3600


def make_field(rows):
    # rows: (id, dra, ddec, pmra, pmdec, tmag, disposition, duplicate_id, has_dr2, in_gaia)
    tic, gaia = [], []
    for tid, dra, ddec, pmra, pmdec, tmag, disp, dup, has_dr2, in_gaia in rows:
        ra, dec = offset(dra, ddec)
        tic.append(dict(ID=tid, ra=ra, dec=dec, pmRA=pmra, pmDEC=pmdec, Tmag=tmag,
                        disposition=disp, duplicate_id=dup,
                        GAIA=int(tid) * 7 if has_dr2 else np.nan,
                        objType="STAR", dstArcSec=np.hypot(dra, ddec)))
        if in_gaia:
            gra, gdec = offset(dra + pmra * 16 / 1000, ddec + pmdec * 16 / 1000)
            gaia.append(dict(source_id=int(tid) * 1000, ra=gra, dec=gdec,
                             phot_g_mean_mag=tmag + 0.5, ruwe=1.05, duplicated_source=0))
    return pd.DataFrame(tic), pd.DataFrame(gaia)


def verdicts(tic, gaia, target):
    return audit_field(tic, gaia, target).set_index("ID")["verdict"]


def test_general_field():
    tic, gaia = make_field([
        ("100", 0, 0, 5, -3, 11.2, "", None, True, True),
        ("101", 4.17, 0, 0, 0, 13.5, "ARTIFACT", None, False, False),
        ("102", 60, 40, 2, 1, 14.0, "", None, True, True),
        ("103", -90, 20, 0, 0, 17.8, "", None, False, False),
        ("104", 30, -100, 400, -250, 12.0, "", None, True, True),
        ("105", 60.8, 40, 0, 0, 14.1, "DUPLICATE", None, False, False),
        ("106", 1.0, 0, 0, 0, 11.4, "", None, False, False),
        ("107", -50, -50, 0, 0, 15.0, "SPLIT", None, True, True),
        ("108", -46, -50, 0, 0, 15.0, "DUPLICATE", "107", False, False),
    ])
    v = verdicts(tic, gaia, "100")
    assert v["100"] == "TARGET"
    assert v["101"] == "PHANTOM"
    assert v["102"] == "OK"
    assert v["103"] == "CHECK"
    assert v["104"] == "OK"  # high proper motion, should still match
    assert v["105"] == "DUPLICATE_COPY"
    assert v["106"] == "CHECK"
    assert v["107"] == "CHECK"
    assert v["108"] == "DUPLICATE_COPY"  # only caught through duplicate_id


def test_target_with_twin():
    # same setup as TOI-7136.01
    tic, gaia = make_field([
        ("200", 0.36, 0, 0, 0, 13.30, "SPLIT", None, True, True),
        ("201", 0, 0, 0, 0, 13.30, "DUPLICATE", None, True, False),
        ("202", 80, 10, 0, 0, 15.0, "", None, True, True),
    ])
    audited = audit_field(tic, gaia, "200")
    v = audited.set_index("ID")["verdict"]
    s = summarize("200", audited)
    assert v["200"] == "TARGET"
    assert v["201"] == "DUPLICATE_COPY"
    assert s["copy_of_target_present"]
    assert s["n_dupcopy_within_1px"] == 1


def test_target_marked_duplicate():
    tic, gaia = make_field([
        ("300", 0, 0, 0, 0, 12.0, "DUPLICATE", "301", True, True),
        ("301", 0.3, 0, 0, 0, 12.0, "", None, True, False),
    ])
    v = verdicts(tic, gaia, "300")
    assert v["300"] == "TARGET"
    assert v["301"] == "CHECK"


if __name__ == "__main__":
    tests = [test_general_field, test_target_with_twin, test_target_marked_duplicate]
    for t in tests:
        t()
        print(f"{t.__name__}: passed")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
