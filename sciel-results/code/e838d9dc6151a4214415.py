def construct():
    p = 1009

    # Precompute a root for each non-zero quadratic residue modulo p
    # roots[res] = smallest positive r in 1..p-1 such that r*r % p == res, or 0 if not a QR
    roots = [0] * p
    for r in range(1, p):
        res = (r * r) % p
        if roots[res] == 0:
            roots[res] = r

    # Linear scan from s=1 upward to find the minimum s
    # where s, s+1, s+2, s+3, s+4 are all non-zero QRs
    for s in range(1, p - 4):
        if (roots[s] != 0 and roots[s+1] != 0 and roots[s+2] != 0 and
            roots[s+3] != 0 and roots[s+4] != 0):
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }

    raise RuntimeError("No solution found")
