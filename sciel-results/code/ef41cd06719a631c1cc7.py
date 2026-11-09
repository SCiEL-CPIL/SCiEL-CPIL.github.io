"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def _is_square(x):
    if x < 0:
        return False
    r = x.bit_length()
    r = 1 << ((r + 1) // 2)
    while True:
        r2 = (r + x // r) // 2
        if r2 >= r:
            break
        r = r2
    return r * r == x

def _valid_fallback(n, hi):
    # 210 * (210 * s^2) = (210*s)^2 is a perfect square.
    for s in range(2, 500):
        val = n * s * s
        if val > n and val <= hi:
            out = [int(val)]
            if _is_square(n * val):
                return out
    return [int(n * 4)]

def construct():
    deadline = time.time() + 52.0
    n = P['n']
    hi = P['maximum_window']
    lo = n + 1

    parities = {}
    for v in range(lo, hi + 1):
        x = v
        p = 2
        bits = set()
        while p * p <= x:
            if x % p == 0:
                c = 0
                while x % p == 0:
                    x //= p
                    c += 1
                if c & 1:
                    bits.add(p)
            p = 3 if p == 2 else p + 2
        if x > 1:
            bits.add(x)
        parities[v] = frozenset(bits)

    target = parities[n]

    def build_basis(upto):
        basis = []
        reprs = []
        for v in range(lo, upto + 1):
            vec = parities[v]
            rep = frozenset((v,))
            for i, b in enumerate(basis):
                inter = b & vec
                if inter:
                    pivot = max(inter)
                    if vec and pivot == max(vec):
                        vec = vec ^ b
                        rep = rep ^ reprs[i]
                        if not vec:
                            break
            if not vec:
                continue
            indep = True
            for b in basis:
                if b & vec:
                    indep = False
                    break
            if indep:
                basis.append(vec)
                reprs.append(rep)
        return basis, reprs

    def reduce_target(basis, reprs, tgt):
        vec = tgt
        rep = frozenset()
        for i, b in enumerate(basis):
            inter = b & vec
            if inter:
                pivot = max(inter)
                if vec and pivot == max(vec):
                    vec = vec ^ b
                    rep = rep ^ reprs[i]
        if not vec and rep:
            return rep
        return None

    best_M = None
    a, b = lo, hi
    while a <= b:
        if time.time() > deadline - 12.0:
            break
        m = (a + b) // 2
        basis, reprs = build_basis(m)
        if reduce_target(basis, reprs, target) is not None:
            best_M = m
            b = m - 1
        else:
            a = m + 1

    if best_M is not None:
        basis, reprs = build_basis(best_M)
        rep = reduce_target(basis, reprs, target)
        if rep is not None:
            vals = [int(x) for x in sorted(rep)]
            if all(lo < x <= best_M for x in vals) and len(vals) >= 1 and len(vals) <= P['maximum_size']:
                prod = n
                for x in vals:
                    prod *= x
                if _is_square(prod):
                    return vals
    return _valid_fallback(n, hi)
# EVOLVE-BLOCK-END
