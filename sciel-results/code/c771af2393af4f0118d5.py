import math

def construct():
    p = 1009
    # Build a boolean array indicating which values in 1..p-1 are quadratic residues
    is_qr = [False] * p
    for x in range(1, p):
        is_qr[(x * x) % p] = True

    # Also build a root lookup: for each residue, store one root
    root_of = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in root_of:
            root_of[r] = x

    # Search for the smallest s in [1, p-5] such that s, s+1, ..., s+4 are all QRs
    for s in range(1, p - 4):
        ok = True
        for i in range(5):
            if not is_qr[s + i]:
                ok = False
                break
        if ok:
            roots = [root_of[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}

    raise RuntimeError("No run of 5 consecutive quadratic residues found")
