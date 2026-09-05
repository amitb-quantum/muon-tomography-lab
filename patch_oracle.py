#!/usr/bin/env python3
"""Fix the oracle column in phase2c_argmax.py.

Bug: threat scenes were scored by the max over ~125 target voxels while benign
scenes used the max over 30k-115k voxels. The max of a larger set is larger,
so threats carried a handicap of 1.42 at 25k rising to 1.86 at 250k -- a
monotonically growing penalty that manufactured the monotone decline.

Fix: every scene gets a PROBE REGION of comparable volume. Threat scenes use
the true target box; benign scenes get a randomly placed box drawn from the
same size distribution, from a dedicated deterministic stream so it is
reproducible and independent of the physics seeds.

Run once: python patch_oracle.py
"""
p = "phase2c_argmax.py"
s = open(p).read()

# 1. import the size list used by the manifest generator
a = "from scene_physics import make_manifest, simulate"
b = "from scene_physics import make_manifest, simulate, TARGET_SIZES_M"
assert a in s, "import line changed"
s = s.replace(a, b)

# 2. build a size-matched probe region for EVERY scene, not just threats
a = """        if m["target"] is not None:
            c = m["target"]["c"].to(dev).double(); h = m["target"]["h"].to(dev).double()
            tmasks.append(((COORD - c).abs() < h).all(-1))
        else:
            tmasks.append(None)"""
b = """        if m["target"] is not None:
            c = m["target"]["c"].to(dev).double(); h = m["target"]["h"].to(dev).double()
        else:
            # size-matched probe region so the oracle max is taken over a
            # comparable volume on both sides. Dedicated RNG stream: this is
            # a diagnostic construct and must not perturb the physics.
            gp = torch.Generator().manual_seed(770_000 + si)
            sz = TARGET_SIZES_M[int(torch.rand(1, generator=gp).item()
                                    * len(TARGET_SIZES_M))]
            lim = side/2 - sz/2 - 0.02
            c = (torch.rand(3, generator=gp)*(2*lim) - lim).to(dev).double()
            h = torch.full((3,), sz/2).to(dev).double()
        tmasks.append(((COORD - c).abs() < h).all(-1))"""
assert a in s, "probe-region block changed"
s = s.replace(a, b)

# 3. oracle score: probe max on BOTH sides. No more global-max fallback.
a = """        ao = float("nan")
        oo = [omax[b][i] for i in T if omax[b][i] == omax[b][i]]
        if len(oo) > 4:
            sc = [omax[b][i] if labels[i] == 1 else gmax[b][i]
                  for i in range(args.n_scenes)]
            # oracle for threats vs global max for benigns: the benign side has
            # no target, so its best available null is the unrestricted max
            ao = roc_auc(sc, labels)"""
b = """        ao = float("nan")
        oo = [omax[b][i] for i in range(args.n_scenes)
              if omax[b][i] == omax[b][i]]
        if len(oo) > 20:
            # probe-region max on BOTH sides: comparable search volumes
            lo = min(oo) - 1.0
            sc = [omax[b][i] if omax[b][i] == omax[b][i] else lo
                  for i in range(args.n_scenes)]
            ao = roc_auc(sc, labels)"""
assert a in s, "oracle scoring block changed"
s = s.replace(a, b)

# 4. hit rate is only meaningful where a real target exists
a = '                tm = tmasks[si]\n                if tm is not None:'
b = '                tm = tmasks[si]\n                if tm is not None:'
assert a in s

# 5. report probe-volume parity so the fix is visible in the output
a = '''    print(f"{'budget':>9} {'AUC glob':>9} {'AUC oracle':>11} {'argmax-in-tgt':>14} "
          f"{'G@argmax':>10} {'edge dist':>10}")'''
b = '''    pv = torch.tensor([float(t.double().sum()) for t in tmasks if t is not None])
    pvT = torch.tensor([float(tmasks[i].double().sum()) for i in T])
    pvN = torch.tensor([float(tmasks[i].double().sum()) for i in N])
    print(f"\\nprobe-region volume (voxels): threat median {float(pvT.median()):.0f}, "
          f"benign median {float(pvN.median()):.0f}  <- must be comparable")
    print(f"\\n{'budget':>9} {'AUC glob':>9} {'AUC oracle':>11} {'argmax-in-probe':>16} "
          f"{'G@argmax':>10} {'edge dist':>10}")'''
assert a in s, "header block changed"
s = s.replace(a, b)

a = '''              f"{ao:11.3f} {sum(hit[b][i] for i in T)/max(len(T),1):14.3f} "'''
b = '''              f"{ao:11.3f} {sum(hit[b][i] for i in T)/max(len(T),1):16.3f} "'''
assert a in s, "row format changed"
s = s.replace(a, b)

open(p, "w").write(s)
print("patched phase2c_argmax.py: oracle now uses size-matched probe regions")
