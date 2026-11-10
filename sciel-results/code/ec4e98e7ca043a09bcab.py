"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before the 90s kill
    MAXM = 128

    # Two residue classes (r1,m1),(r2,m2) are disjoint iff
    # gcd(m1,m2) does NOT divide (r1 - r2), i.e. (r1 - r2) % gcd(m1,m2) != 0.
    def disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0

    def is_valid(pairs):
        n = len(pairs)
        for i in range(n):
            r_i, m_i = pairs[i]
            for j in range(i + 1, n):
                r_j, m_j = pairs[j]
                if not disjoint(r_i, m_i, r_j, m_j):
                    return False
        return True

    def can_add(pairs, m):
        # Smallest r in [0, m) disjoint from every existing class, else None.
        pre = [(math.gcd(m, m_p), r_p) for (r_p, m_p) in pairs]
        for r in range(m):
            ok = True
            for g, r_p in pre:
                if (r - r_p) % g == 0:
                    ok = False
                    break
            if ok:
                return r
        return None

    def seed_pow2():
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

    best = seed_pow2()

    all_moduli = list(range(2, MAXM + 1))

    def bitcount(x):
        c = 0
        while x:
            c += x & 1
            x >>= 1
        return c

    random.seed(20240613)
    orders = [
        list(range(2, MAXM + 1)),
        list(range(MAXM, 1, -1)),
        sorted(all_moduli, key=lambda m: -m.bit_length()),
        sorted(all_moduli, key=lambda m: -bitcount(m)),
        [m for m in range(2, MAXM + 1) if m % 2 == 0] +
        [m for m in range(2, MAXM + 1) if m % 2 == 1],
        [m for m in range(2, MAXM + 1) if m % 3 == 0] +
        [m for m in range(2, MAXM + 1) if m % 3 != 0],
    ]
    for _ in range(12):
        o = list(all_moduli)
        random.shuffle(o)
        orders.append(o)

    def build(order, seed_pairs=None):
        pairs = list(seed_pairs) if seed_pairs else []
        used = set(m for _, m in pairs)
        for m in order:
            if m in used:
                continue
            if time.time() > deadline - 5:
                break
            r = can_add(pairs, m)
            if r is not None:
                pairs.append([r, m])
                used.add(m)
        return pairs

    for order in orders:
        for sp in (None, seed_pow2()):
            cand = build(order, sp)
            if is_valid(cand) and len(cand) > len(best):
                best = cand

    # Local improvement: repeatedly add any still-feasible modulus.
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        used = set(m for _, m in best)
        for m in range(2, MAXM + 1):
            if m in used:
                continue
            r = can_add(best, m)
            if r is not None:
                best.append([r, m])
                used.add(m)
                improved = True
                break

    if not is_valid(best):
        best = seed_pow2()
    if not best:
        best = [[0, 2]]
    return best
# EVOLVE-BLOCK-END
