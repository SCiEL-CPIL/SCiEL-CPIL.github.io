"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # explicit safety margin under the 90s kill
    MAXM = 128
    moduli = list(range(2, MAXM + 1))
    primes = [m for m in moduli if all(m % k for k in range(2, int(m ** 0.5) + 1))]
    prime_set = set(primes)
    comps = [m for m in moduli if m not in prime_set]

    def valid_assignment(assignment, m, r):
        for (rj, mj) in assignment:
            g = _gcd(m, mj)
            if r % g == rj % g:
                return False
        return True

    def find_residue(m, assignment):
        for r in range(m):
            if valid_assignment(assignment, m, r):
                return r
        return None

    def greedy(order):
        assignment = []
        for m in order:
            r = find_residue(m, assignment)
            if r is not None:
                assignment.append((r, m))
        return assignment

    best = None
    best_count = 0

    orderings = [primes + comps, primes + sorted(comps, reverse=True)]
    rng = random.Random(12345)
    for _ in range(300):
        c = comps[:]
        rng.shuffle(c)
        orderings.append(primes + c)

    for order in orderings:
        if time.time() > deadline:
            break
        a = greedy(order)
        if len(a) > best_count:
            best_count = len(a)
            best = a

    if best is None or best_count < len(primes):
        best = [(0, p) for p in primes]
        best_count = len(primes)

    # Local improvement: keep adding any modulus that fits
    used = set(m for (r, m) in best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for m in moduli:
            if m in used:
                continue
            r = find_residue(m, best)
            if r is not None:
                best.append((r, m))
                used.add(m)
                improved = True
                break

    return [[r, m] for (r, m) in best]
# EVOLVE-BLOCK-END
