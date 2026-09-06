#!/usr/bin/env python3
"""Phase 4A-NF: held-out cargo with a 1 mrad / 1 mm likelihood floor."""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch
import phase3_ideal_observer as p3
import phase4_heldout_cargo as p4

SIGMA_THETA_MRAD = 1.0
SIGMA_X_MM = 1.0
SIGMA_THETA = 1e-3
SIGMA_X = 1e-3
LOG2PI = math.log(2.0 * math.pi)


def marginal_event_logpdf_with_floor(J0, J1, J2, obs, pgrid, logw):
    J0 = J0.clamp_min(1e-14)
    logf = (1.0 + 0.038 * torch.log(J0)).clamp_min(0.0)
    lf2 = logf * logf
    k00 = (lf2 * J0).clamp_min(1e-18)
    k01 = lf2 * J1
    k11 = (lf2 * J2).clamp_min(1e-18)

    tx, ty, dx, dy = obs["tx"], obs["ty"], obs["dx"], obs["dy"]
    while tx.ndim < J0.ndim:
        tx, ty, dx, dy = tx.unsqueeze(0), ty.unsqueeze(0), dx.unsqueeze(0), dy.unsqueeze(0)

    s = (13.6 / pgrid) ** 2
    A = s * k00[..., None] + SIGMA_THETA**2
    B = s * k01[..., None]
    C = s * k11[..., None] + SIGMA_X**2
    det = (A * C - B * B).clamp_min(1e-30)

    tx, ty, dx, dy = tx[..., None], ty[..., None], dx[..., None], dy[..., None]
    qx = (C*tx*tx - 2*B*tx*dx + A*dx*dx) / det
    qy = (C*ty*ty - 2*B*ty*dy + A*dy*dy) / det
    lp = -2.0*LOG2PI - torch.log(det) - 0.5*(qx+qy) + logw
    return torch.logsumexp(lp, dim=-1)


def protocol_hash():
    p = Path("PHASE4_NOISEFLOOR_PROTOCOL.md")
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def save_path_from_argv():
    if "--save" in sys.argv:
        i = sys.argv.index("--save")
        if i+1 < len(sys.argv):
            return sys.argv[i+1]
    return "phase4a_heldout_cargo.npz"


def annotate(path):
    p = Path(path)
    if not p.exists():
        return
    with np.load(p, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["metadata_json"]))
    meta["phase"] = "4A-NF"
    meta["likelihood_sigma_theta_mrad"] = SIGMA_THETA_MRAD
    meta["likelihood_sigma_x_mm"] = SIGMA_X_MM
    meta["likelihood_floor_only"] = True
    meta["protocol_sha256"] = protocol_hash()
    arrays["metadata_json"] = np.array(json.dumps(meta))
    np.savez_compressed(p, **arrays)

    s = meta["results"]["static"]["auc"]
    t = meta["results"]["two_view"]["auc"]
    best = max(s, t)
    print("\nPHASE 4A-NF DECISION")
    if best < 0.60:
        print("COLLAPSE PERSISTS: both blind AUCs < 0.60. Close project.")
    elif best >= 0.70:
        print("MATERIAL RECOVERY: run exactly one 10,000-track check with same floors, then close.")
    else:
        print("INTERMEDIATE: report sensitivity to likelihood regularization and close; no tuning branch.")
    print("noise-floor protocol sha256:", meta["protocol_sha256"])


def main():
    print("PHASE 4A-NF: likelihood-floor sensitivity")
    print("sigma_theta = 1.0 mrad")
    print("sigma_x     = 1.0 mm")
    print("NOTE: event measurements remain ideal/noiseless; this is a likelihood floor.")
    p3.marginal_event_logpdf = marginal_event_logpdf_with_floor
    out = save_path_from_argv()
    p4.main()
    annotate(out)


if __name__ == "__main__":
    main()
