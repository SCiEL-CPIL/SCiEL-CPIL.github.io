"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50  # safety margin before 60s / 90s kill
    N = 512

    # Build all numbers of the form 2^i * 3^j <= 512.
    # For any two such numbers a=2^i*3^j, b=2^k*3^l,
    # lcm(a,b)=2^max(i,k)*3^max(j,l) <= max(a,b) <= 512.
    S = set()
    p = 1
    while p <= N:
        q = p
        while q <= N:
            S.add(q)
            q *= 3
        p *= 2

    result = sorted(S)
    if time.time() > deadline:
        return result

    # Optional: try to greedily add any remaining number x in [1,512]
    # that satisfies lcm(x, s) <= 512 for all s in S.
    from math import gcd
    added = True
    while added:
        added = False
        for x in range(1, N + 1):
            if x in S:
                continue
            ok = True
            for s in S:
                l = s // gcd(s, x) * x
                if l > N:
                    ok = False
                    break
            if ok:
                S.add(x)
                added = True
        if time.time() > deadline:
            break

    return sorted(S)
# EVOLVE-BLOCK-END
