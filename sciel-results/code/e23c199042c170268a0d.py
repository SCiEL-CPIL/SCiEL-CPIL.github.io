import time

def construct():
    p = 1009
    deadline = time.monotonic() + 85  # safety margin before 90s kill

    # Precompute a root for each quadratic residue modulo p
    # roots[res] = smallest positive r in 1..p-1 such that r*r % p == res, or 0
    roots = [0] * p
    for r in range(1, p):
        res = (r * r) % p
        if roots[res] == 0:
            roots[res] = r

    # is_res[x] True iff x is a non-zero QR mod p
    is_res = [False] * p
    for res in range(1, p):
        if roots[res] != 0:
            is_res[res] = True

    # Search for minimum s in 1..1004 such that s, s+1, s+2, s+3, s+4 are all QRs
    for s in range(1, 1005):
        if time.monotonic() > deadline:
            break
        if is_res[s] and is_res[s+1] and is_res[s+2] and is_res[s+3] and is_res[s+4]:
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }

    # Fallback: should not happen for p=1009, m=5
    raise RuntimeError("No solution found")
