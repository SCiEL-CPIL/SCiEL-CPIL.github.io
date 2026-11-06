"""Optimize a set of 20 distinct positive integers <=10000 to minimize
max(|A+A|, |A*A|) with diagonal pairs allowed."""
import time
import random

def _sumset_size(A):
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai + A[j])
    return len(s)

def _prodset_size(A):
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai * A[j])
    return len(s)

def _score(A):
    return max(_sumset_size(A), _prodset_size(A))

def construct():
    start = time.time()
    deadline = start + 85.0   # safety margin before 90s kill
    n = 20
    maxv = 10000

    # Deterministic good starting point: powers of 2 (small product set)
    def make_powers(base):
        A = []
        x = base
        while len(A) < n and x <= maxv:
            A.append(x)
            x *= base
        return A

    # Candidate initial sets
    inits = []
    for b in (2, 3):
        A = make_powers(b)
        if len(A) == n:
            inits.append(list(A))

    # Also a mixed geometric start
    def make_geo2():
        # 10 powers of 2 and 10 powers of 3 interleaved to keep product set small
        A = []
        for e in range(0, 10):
            A.append(2 ** e)
        for e in range(1, 11):
            v = 3 ** e
            if v <= maxv:
                A.append(v)
        # ensure 20 distinct
        A = sorted(set(A))
        while len(A) < n:
            A.append(maxv - (n - len(A)))
        A = sorted(set(A))[:n]
        return A
    inits.append(make_geo2())

    best = None
    best_score = float('inf')
    for A in inits:
        if len(A) == n and all(1 <= x <= maxv for x in A) and len(set(A)) == n:
            sc = _score(A)
            if sc < best_score:
                best_score = sc
                best = list(A)

    rng = random.Random(12345)

    def local_search(A, time_budget):
        A = list(A)
        cur = _score(A)
        n = len(A)
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            # random order of pairs
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                if time.time() >= time_budget:
                    break
                for j in range(n):
                    if i == j:
                        continue
                    old = A[j]
                    # propose replacing A[j] with a new value
                    # candidate values: nearby to keep structure, or random
                    for cand in _candidates(old, rng, maxv, A):
                        if cand == old:
                            continue
                        if cand in A:
                            continue
                        A[j] = cand
                        ns = _score(A)
                        if ns < cur:
                            cur = ns
                            improved = True
                            break
                        A[j] = old
                    if improved:
                        break
        return A, cur

    def _candidates(old, rng, maxv, A):
        cands = []
        # multiplicative neighbors (preserve product structure)
        for f in (2, 3, 5):
            cands.append(old * f)
            cands.append(old // f if old % f == 0 else None)
        # additive neighbors
        for d in (-1, 1, -2, 2, -3, 3):
            cands.append(old + d)
        # random
        for _ in range(3):
            cands.append(rng.randint(1, maxv))
        # powers of 2 / 3 near old
        for b in (2, 3):
            e = max(1, int(round(__import__('math').log(old, b))))
            for ee in (e - 1, e, e + 1):
                if ee >= 1:
                    cands.append(b ** ee)
        out = []
        seen = set()
        for c in cands:
            if c is None:
                continue
            if not (1 <= c <= maxv):
                continue
            if c in seen:
                continue
            seen.add(c)
            out.append(c)
        return out

    # Run local search from best initial, with a few random restarts if time
    t_end = deadline - 3.0
    A, sc = local_search(best, t_end)
    if sc < best_score:
        best_score = sc
        best = A

    # If we still have time, try a couple of random restarts
    if time.time() < deadline - 8.0:
        for restart in range(3):
            if time.time() >= deadline - 8.0:
                break
            # random start biased toward small numbers (small product set)
            A = sorted(rng.sample(range(1, maxv + 1), n))
            A2, sc2 = local_search(A, deadline - 8.0)
            if sc2 < best_score:
                best_score = sc2
                best = A2

    return list(best)
