#!/usr/bin/env python3
"""Expose MIN_COUNT as a CLI flag. Run once: python patch_harness.py"""
p = "scene_harness.py"
s = open(p).read()

a = '    ap.add_argument("--nvox", type=int, default=50)'
b = a + '\n    ap.add_argument("--min-count", type=int, default=2)'
assert a in s and b not in s, "arg already patched or file changed"
s = s.replace(a, b)

c = '    args = ap.parse_args()\n\n    dev = torch.device(args.device)'
d = ('    args = ap.parse_args()\n\n    global MIN_COUNT\n'
     '    MIN_COUNT = args.min_count\n\n    dev = torch.device(args.device)')
assert c in s, "main() body changed"
s = s.replace(c, d)

e = 'def main():\n    ap = argparse.ArgumentParser()'
s = s.replace(e, 'def main():\n    ap = argparse.ArgumentParser()')

# report the actual value in use
f = 'print(f"statistic: mask-normalized {KERNEL}^3 filter, "\n          f"min_support={MIN_KERNEL_SUPPORT}, min_count={MIN_COUNT}\\n")'
assert f in s, "banner changed"

open(p, "w").write(s)
print("patched scene_harness.py -> --min-count available")
