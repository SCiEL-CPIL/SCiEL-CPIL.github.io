"""Suite task: maximize size of a set in 2..256 with no a | b*c for distinct a,b,c."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

P = {'n': 256}


def _prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _precompute(maxn):
    pf = [set() for _ in range(maxn + 1)]
    for i in range(2, maxn + 1):
        pf[i] = _prime_factors(i)
    return pf


def _valid_add(pf, S, x):
    # Check that no existing a in S divides x*b for any other b in S (b != x, b != a)
    fx = pf[x]
    for a in S:
        if a == x:
            continue
        fa = pf[a]
        for b in S:
            if b == x or b == a:
                continue
            # a | x*b  <=>  every prime factor of a divides x*b
            fb = pf[b]
            ok = True
            for p in fa:
                if p not in fx and p not in fb:
                    ok = False
                    break
            if ok:
                return False
    return True


def _build_greedy(pf, maxn):
    # Try several orderings, keep the best resulting set
    best = None
    orderings = [
        list(range(2, maxn + 1)),
        list(range(maxn, 1, -1)),
        sorted(range(2, maxn + 1), key=lambda v: (len(pf[v]), v)),
        sorted(range(2, maxn + 1), key=lambda v: (len(pf[v]), -v)),
    ]
    for order in orderings:
        S = []
        for x in order:
            if _valid_add(pf, S, x):
                S.append(x)
        if best is None or len(S) > len(best):
            best = S
    return best


def _swap_search(pf, S, candidates, deadline, iters):
    # Local search: try adding a candidate and removing one element to stay valid
    S = list(S)
    cand = [c for c in candidates if c not in S]
    cand.sort()
    improved = True
    it = 0
    while improved and it < iters and time.time() < deadline:
        improved = False
        it += 1
        # Try each candidate
        for c in cand:
            if time.time() > deadline:
                return S
            if c in S:
                continue
            # Try adding c; if invalid, try removing one element at a time
            # First check if c alone is addable
            if _valid_add(pf, S, c):
                S.append(c)
                improved = True
                continue
            # Otherwise, try removing one element r so that S - {r} + {c} is valid
            for r in S:
                if time.time() > deadline:
                    return S
                newS = [y for y in S if y != r]
                newS.append(c)
                if _valid_add(pf, newS, c):
                    S = newS
                    improved = True
                    break
    return S


def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # safety margin vs 60s/90s limits
    maxn = P['n']
    pf = _precompute(maxn)

    S = _build_greedy(pf, maxn)

    # Add-only pass (fast, often completes)
    if time.time() < deadline:
        S = _swap_search(pf, S, list(range(2, maxn + 1)), deadline, iters=20)

    # If still under deadline, do a few more swap rounds with fresh candidate order
    if time.time() < deadline - 5.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            S = _swap_search(pf, S, list(range(maxn, 1, -1)), deadline, iters=10)

    # Ensure Python ints in a list
    return [int(x) for x in S]
# EVOLVE-BLOCK-END
