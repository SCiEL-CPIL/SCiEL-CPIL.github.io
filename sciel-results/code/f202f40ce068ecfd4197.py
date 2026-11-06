import time
import numpy as np

# EVOLVE-BLOCK-START
def _primes_upto(n):
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:: i] = False
    return [int(p) for p in np.flatnonzero(sieve)]

def _coverage_length(covered, cap=100000):
    if not covered.any():
        return 0
    first = int(np.flatnonzero(covered)[0])
    if first > 1:
        return 0
    last = int(np.flatnonzero(covered)[-1])
    if last > cap:
        last = cap
    seg = covered[first:last + 1]
    if seg.size == 0:
        return 0
    if not seg.all():
        # find first zero
        z = np.flatnonzero(~seg)
        if z.size:
            return first + int(z[0]) - 1
    return last

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill
    primes = _primes_upto(1000)
    n = len(primes)
    idx = {p: i for i, p in enumerate(primes)}
    cap = 100000
    covered = np.zeros(cap + 1, dtype=bool)
    used = np.zeros(n, dtype=bool)
    budget = 1.0
    spent = 0.0

    def time_ok():
        return time.time() < deadline

    # Phase 1: small primes (p <= 37) with greedy residue choice
    for p in primes:
        if p > 37:
            break
        if not time_ok():
            break
        best_r = 0
        best_c = -1
        for r in range(p):
            m = (np.arange(1, cap + 1) - r) % p
            c = int(np.sum(~covered & (m == 0)))
            if c > best_c:
                best_c = c
                best_r = r
        i = idx[p]
        m = (np.arange(1, cap + 1) - best_r) % p
        covered |= (m == 0)
        used[i] = True
        spent += 1.0 / p

    # Phase 2: greedy set-cover with larger primes, bounded by budget
    remaining = primes[primes > 37]
    for p in remaining:
        if not time_ok():
            break
        if spent + 1.0 / p > budget + 1e-12:
            continue
        best_r = 0
        best_c = -1
        for r in range(p):
            m = (np.arange(1, cap + 1) - r) % p
            c = int(np.sum(~covered & (m == 0)))
            if c > best_c:
                best_c = c
                best_r = r
        if best_c <= 0:
            continue
        i = idx[p]
        m = (np.arange(1, cap + 1) - best_r) % p
        covered |= (m == 0)
        used[i] = True
        spent += 1.0 / p
        L = _coverage_length(covered)
        if L >= 999 and L < 100000:
            # we already reach the target; keep going to extend coverage
            pass

    # Build result
    result = []
    for i, p in enumerate(primes):
        if used[i]:
            # recover residue: we stored via covered, but we need r.
            # We didn't store r per prime; recompute from coverage is hard.
            # Instead, recompute residues deterministically below.
            pass

    # Recompute: we need actual residues. Redo storage properly.
    # (Faster to have stored them; but to keep it simple, rebuild from scratch
    #  with the same deterministic choices.)
    return _build_with_storage(primes, idx, cap, deadline)

def _build_with_storage(primes, idx, cap, deadline):
    t0 = time.time()
    covered = np.zeros(cap + 1, dtype=bool)
    used = np.zeros(len(primes), dtype=bool)
    residues = {}
    spent = 0.0
    budget = 1.0

    def time_ok():
        return time.time() < deadline

    for p in primes:
        if p > 37:
            break
        if not time_ok():
            break
        best_r = 0
        best_c = -1
        for r in range(p):
            m = (np.arange(1, cap + 1) - r) % p
            c = int(np.sum(~covered & (m == 0)))
            if c > best_c:
                best_c = c
                best_r = r
        i = idx[p]
        m = (np.arange(1, cap + 1) - best_r) % p
        covered |= (m == 0)
        used[i] = True
        residues[p] = best_r
        spent += 1.0 / p

    for p in primes:
        if p <= 37:
            continue
        if not time_ok():
            break
        if spent + 1.0 / p > budget + 1e-12:
            continue
        best_r = 0
        best_c = -1
        for r in range(p):
            m = (np.arange(1, cap + 1) - r) % p
            c = int(np.sum(~covered & (m == 0)))
            if c > best_c:
                best_c = c
                best_r = r
        if best_c <= 0:
            continue
        i = idx[p]
        m = (np.arange(1, cap + 1) - best_r) % p
        covered |= (m == 0)
        used[i] = True
        residues[p] = best_r
        spent += 1.0 / p

    result = [[p, residues[p]] for p in primes if used[idx[p]]]
    return result
# EVOLVE-BLOCK-END
