#!/usr/bin/env python3
"""
Patch phase2d_calibrated.py for:
  1) Float32/Float64 assignment crash in empirical tail calibration.
  2) Saturation detection using the WINNING voxel's stratum pool, rather than
     the largest pool across all strata.

No experiment parameters, bin counts, alpha, G_MIN, kernel, folds, or scoring
architecture are changed.

Usage:
    python patch_phase2d.py /home/manager/muon/phase2d_calibrated.py
"""

import sys
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python patch_phase2d.py /path/to/phase2d_calibrated.py"
        )

    path = Path(sys.argv[1]).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")

    text = path.read_text()

    old_dtype = '                Tf = torch.full_like(Ff, float("-inf"))\n'
    new_dtype = '''                Tf = torch.full(
                    Ff.shape,
                    float("-inf"),
                    device=Ff.device,
                    dtype=torch.float64,
                )
'''

    old_sat = '''                cap = float(torch.log10(torch.tensor(
                    max(x.numel() for x in sortd if x is not None) + 1.0)))
                mx = float(Tf.max())
                if mx >= cap - 1e-9:
                    sat[b] += 1
                # tie-break saturated scenes by the raw filtered max
                cal[b][si] = mx + 1e-6*raw[b][si]
                fl = int(torch.argmax(Tf))
                i3 = (fl//(nv*nv), (fl//nv) % nv, fl % nv)
                aG[b][si] = float(G[b][i3]); aE[b][si] = float(edged[i3])
'''
    new_sat = '''                cal_fin = torch.isfinite(Tf)
                if not cal_fin.any():
                    # No calibrated voxel in this scene/fold. Leave the
                    # explicit no-evidence score initialized above.
                    continue

                mx = float(Tf[cal_fin].max())
                fl = int(torch.argmax(Tf))
                i3 = (fl//(nv*nv), (fl//nv) % nv, fl % nv)

                # Saturation must be judged against the empirical pool for
                # the ACTUAL WINNING voxel's stratum, not the largest pool.
                swin = int(st[i3])
                if swin < 0 or sortd[swin] is None:
                    continue
                Nwin = sortd[swin].numel()
                cap_win = float(torch.log10(torch.tensor(
                    Nwin + 1.0, dtype=torch.float64, device=dev)))
                is_sat = mx >= cap_win - 1e-9
                if is_sat:
                    sat[b] += 1

                # Raw-field tie break only matters when this winning stratum
                # is empirically saturated.
                cal[b][si] = mx + (1e-6*raw[b][si] if is_sat else 0.0)

                aG[b][si] = float(G[b][i3]); aE[b][si] = float(edged[i3])
'''

    if old_dtype not in text:
        raise SystemExit(
            "ERROR: dtype target block not found. File may already be patched "
            "or differs from the supplied version."
        )
    if old_sat not in text:
        raise SystemExit(
            "ERROR: saturation target block not found. File may already be "
            "patched or differs from the supplied version."
        )

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(text)

    text = text.replace(old_dtype, new_dtype, 1)
    text = text.replace(old_sat, new_sat, 1)

    # Syntax-check before overwriting the source.
    compile(text, str(path), "exec")
    path.write_text(text)

    print(f"PATCHED: {path}")
    print(f"BACKUP : {backup}")
    print("Syntax check: PASS")
    print()
    print("Now run:")
    print(f"  python {path.name} --n-scenes 400 --save phase2d_dev.npz")

if __name__ == "__main__":
    main()
