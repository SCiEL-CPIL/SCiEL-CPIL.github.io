import time
import math
import random
from functools import lru_cache

_DEADLINE = 145.0  # safety margin under the 150s kill
_T0 = time.time()

# primes up to 67 (product of first 18 = 3.2e19 > 1e18, so k<=17 feasible)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]

# exponent caps so that the product never overflows 1e18 when all caps are used
def _exp_cap(p, k):
    # largest e with p^e * (product of k-1 smallest other primes) <= 1e18
    others = 1
    for q in PRIMES:
        if q != p:
            others *= q
            if others > 10**18:
                break
    if others > 10**18:
        others = 10**18
    limit = 10**18 // others
    e = 0
    cur = 1
    while cur * p <= limit:
        cur *= p
        e += 1
    return e

def _score_exponents(exps):
    """Estimate tau_perp for n = prod p_i^e_i using the (e_i+1) grid."""
    m = len(exps)
    if m == 0:
        return 0
    # grid index -> (prime_idx, exp)
    grid = []
    for pi, e in enumerate(exps):
        for j in range(e + 1):
            grid.append((pi, j))
    # sort by log value
    grid.sort(key=lambda x: x[0] * math.log(PRIMES[x[0]]) + x[1] * math.log(PRIMES[x[0]]))
    score = 0
    for a in range(len(grid) - 1):
        pi, ei = grid[a]
        pj, ej = grid[a + 1]
        if pi != pj:
            score += 1
    return score

def _n_from_exponents(exps):
    n = 1
    for pi, e in enumerate(exps):
        if e:
            n *= PRIMES[pi] ** e
    return n

def _local_polish(exps):
    """Hill-climb on exponent adjustments (increase/decrease one prime)."""
    exps = list(exps)
    best = _score_exponents(exps)
    improved = True
    guard = 0
    while improved and guard < 200:
        improved = False
        guard += 1
        for pi in range(len(exps)):
            # try +1
            if exps[pi] < 60:
                exps[pi] += 1
                if _n_from_exponents(exps) <= 10**18:
                    s = _score_exponents(exps)
                    if s > best:
                        best = s
                        improved = True
                        continue
                exps[pi] -= 1
            # try -1
            if exps[pi] > 0:
                exps[pi] -= 1
                s = _score_exponents(exps)
                if s > best:
                    best = s
                    improved = True
                    continue
                exps[pi] += 1
    return exps, best

def construct():
    global _T0, _DEADLINE
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = _score_exponents([1] * 8)

    def _consider(exps):
        nonlocal best_n, best_score
        n = _n_from_exponents(exps)
        if n < 2 or n > 10**18:
            return
        s = _score_exponents(exps)
        if s > best_score:
            best_score = s
            best_n = n

    # Phase 1: DFS over monotone exponent vectors (e1>=e2>=...>=ek), k up to 17.
    def dfs(pi, max_e, exps):
        if time.time() > _DEADLINE - 2:
            return
        if pi == len(PRIMES):
            _consider(exps)
            return
        p = PRIMES[pi]
        cap = min(max_e, _exp_cap(p, pi + 1))
        # try exponents from cap down to 0
        for e in range(cap, -1, -1):
            exps.append(e)
            dfs(pi + 1, e, exps)
            exps.pop()

    # We cannot DFS all k; instead iterate k and do bounded DFS with pruning by product.
    # Use a smarter recursive that tracks current product to prune.
    def dfs2(pi, k, max_e, exps, prod):
        if time.time() > _DEADLINE - 2:
            return
        if pi == k:
            _consider(exps)
            return
        p = PRIMES[pi]
        cap = min(max_e, _exp_cap(p, k))
        for e in range(cap, -1, -1):
            newprod = prod * (p ** e)
            if newprod > 10**18:
                continue
            exps.append(e)
            dfs2(pi + 1, k, e, exps, newprod)
            exps.pop()

    # Start with small k and grow, but DFS space explodes; use a time-sliced approach.
    # Try k = number of primes used. For each k, do DFS but stop early if too slow.
    # We'll do a randomized + structured hybrid to stay within budget.

    # First, a quick structured pass for moderate k with a step limit.
    for k in range(2, 18):
        if time.time() > _DEADLINE - 5:
            break
        exps = []
        # DFS but cap total nodes via time check inside
        dfs2(0, k, 60, exps, 1)

    # Phase 2: local polish of the current best exponent vector and random restarts.
    # Recover exponents of best_n.
    def _exponents_of(n):
        ex = []
        for p in PRIMES:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            ex.append(e)
        return ex

    if time.time() < _DEADLINE - 5:
        cur_ex = _exponents_of(best_n)
        cur_ex, cur_s = _local_polish(cur_ex)
        if cur_s > best_score:
            best_score = cur_s
            best_n = _n_from_exponents(cur_ex)

    # Randomized search: pick k, assign random monotone-ish exponents, polish.
    rng = random.Random(12345)
    while time.time() < _DEADLINE - 3:
        k = rng.randint(3, 17)
        # random exponents, then make monotone non-increasing by sorting desc
        base = []
        for i in range(k):
            cap = _exp_cap(PRIMES[i], k)
            base.append(rng.randint(0, max(1, cap)))
        base.sort(reverse=True)
        # ensure product <= 1e18
        prod = 1
        for i, e in enumerate(base):
            prod *= PRIMES[i] ** e
            if prod > 10**18:
                # scale down
                scale = 10**18 // prod
                for i2 in range(k):
                    while base[i2] > 0 and PRIMES[i2] ** (base[i2]) > scale:
                        base[i2] -= 1
                        scale = 10**18 // (prod // (PRIMES[i2] ** (base[i2] + 1)) * (PRIMES[i2] ** (base[i2])))
                break
        prod = 1
        for i, e in enumerate(base):
            prod *= PRIMES[i] ** e
        if 2 <= prod <= 10**18:
            polished, s = _local_polish(base)
            n = _n_from_exponents(polished)
            if 2 <= n <= 10**18:
                s2 = _score_exponents(polished)
                if s2 > best_score:
                    best_score = s2
                    best_n = n

    return best_n
