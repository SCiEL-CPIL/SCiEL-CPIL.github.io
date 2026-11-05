"""Rewritten program: parity-forced bipartite search with prime-mask pruning.

We must return two lists A,B of 3 distinct integers in [1,128] such that all
DISTINCT values in the sumset A+B are pairwise coprime, maximising |A+B|.

Key structural insight (parity):
  Pairwise-coprime => at most ONE even number in the whole sumset.
  For a 3x3 additive grid a_i+b_j the number of even sums is
      k*m + (3-k)*(3-m)   (k=#even in A, m=#even in B)
  This equals 1 for NO (k,m) in [0,3]^2, so we must have ZERO even sums.
  That forces (A all odd & B all even) or (A all even & B all odd).
  Hence every sum is ODD and factor 2 never appears; we only need to manage
  odd prime factors.  This halves the search space with no loss.

Search:
  * Precompute an odd-prime-factor bitmask for every integer 1..256.
  * Enumerate A (odd, 3 distinct) and B (even, 3 distinct); the 9 sums must be
    distinct and their masks pairwise disjoint.  We also try the swapped parity
    (A even, B odd).  A greedy + randomized pass finds 9 distinct coprime sums.
"""
import itertools
import random
import time

# ---- deadline guard -------------------------------------------------------
_DEADLINE = time.time() + 80.0   # safety margin before the 90s kill

# ---- prime factor masks (odd primes only) --------------------------------
_MAX = 256
_ODD_PRIMES = [p for p in range(3, _MAX + 1)
               if all(p % q for q in range(3, int(p ** 0.5) + 1, 2))]
_mask = [0] * (_MAX + 1)
for i, p in enumerate(_ODD_PRIMES):
    for m in range(p, _MAX + 1, p):
        _mask[m] |= (1 << i)


def _pairwise_coprime(vals):
    """True if every pair of distinct values in vals is coprime."""
    seen = set()
    for v in vals:
        if v in seen:
            continue
        for u in seen:
            import math
            if math.gcd(u, v) != 1:
                return False
        seen.add(v)
    return True


def _valid(A, B):
    sums = [a + b for a in A for b in B]
    distinct = set(sums)
    # pairwise coprime among distinct values
    d = list(distinct)
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            import math
            if math.gcd(d[i], d[j]) != 1:
                return False
    return True


def _score(A, B):
    return len({a + b for a in A for b in B})


def construct():
    odds = [x for x in range(1, 129) if x % 2 == 1]
    evens = [x for x in range(2, 129) if x % 2 == 0]

    best = ([[1, 7, 13], [4, 10, 16]], 5)  # fallback seed

    def consider(A, B):
        nonlocal best
        if not _valid(A, B):
            return
        s = _score(A, B)
        if s > best[1]:
            best = ([list(A), list(B)], s)
            if best[1] >= 9:
                return True
        return False

    # ---- deterministic greedy: build A, then B to hit 9 distinct coprime sums
    # We want 9 distinct sums all pairwise coprime.  Try to assemble a 3x3 grid.
    # Strategy: pick 9 target odd numbers pairwise coprime, then solve for A,B.
    # Simpler: brute small candidate sets.
    rng = random.Random(12345)

    # Precompute masks for quick pairwise-coprime test of a sumset.
    def masks_disjoint(sums):
        acc = 0
        for s in sums:
            if s < 1 or s > _MAX:
                return False
            if acc & _mask[s]:
                return False
            acc |= _mask[s]
        return True

    # Enumerate A over a reduced odd pool, B over reduced even pool.
    # To keep it fast, cap pool sizes and use deadline guard.
    A_pool = odds
    B_pool = evens

    # Randomized search for a 9-sum solution.
    found = False
    iters = 0
    while time.time() < _DEADLINE and not found:
        iters += 1
        A = rng.sample(A_pool, 3)
        B = rng.sample(B_pool, 3)
        sums = [a + b for a in A for b in B]
        if len(set(sums)) < 9:
            continue
        if masks_disjoint(sums):
            # confirm with exact gcd
            if _valid(A, B):
                best = ([list(A), list(B)], 9)
                found = True
                break
        # also try swapped parity
        A2 = rng.sample(B_pool, 3)
        B2 = rng.sample(A_pool, 3)
        sums2 = [a + b for a in A2 for b in B2]
        if len(set(sums2)) == 9 and masks_disjoint(sums2):
            if _valid(A2, B2):
                best = ([list(A2), list(B2)], 9)
                found = True
                break

    # If randomized didn't find 9, do a focused deterministic sweep over a
    # small curated set of candidate values (fast).
    if best[1] < 9 and time.time() < _DEADLINE:
        small_odd = [x for x in odds if x <= 60]
        small_even = [x for x in evens if x <= 60]
        for A in itertools.combinations(small_odd, 3):
            if time.time() >= _DEADLINE:
                break
            for B in itertools.combinations(small_even, 3):
                sums = [a + b for a in A for b in B]
                if len(set(sums)) == 9 and masks_disjoint(sums):
                    if _valid(A, B):
                        best = ([list(A), list(B)], 9)
                        break
            if best[1] >= 9:
                break

    return best[0]
