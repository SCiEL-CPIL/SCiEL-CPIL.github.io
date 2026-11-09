"""Rewritten program: build conflict graph, then greedy + local-search MIS with deadline guard."""
import time
import random

_N = 256
_T = 90.0  # evaluator wall-clock limit; keep safety margin
_MARGIN = 5.0

# Precompute conflict graph: edge (a,b) if a*b is divisible by a+b
def _build_conflicts():
    conflicts = [set() for _ in range(_N + 1)]
    for a in range(1, _N + 1):
        ca = conflicts[a]
        for b in range(a + 1, _N + 1):
            s = a + b
            if (a * b) % s == 0:
                ca.add(b)
                conflicts[b].add(a)
    return conflicts

_CONFLICTS = _build_conflicts()
_DEG = [len(c) for c in _CONFLICTS]

def _valid(S):
    # S is a set of ints in 1..256; check every distinct pair
    L = sorted(S)
    for i in range(len(L)):
        a = L[i]
        for j in range(i + 1, len(L)):
            b = L[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

def _greedy_init(order):
    S = set()
    for a in order:
        if not (S & _CONFLICTS[a]):
            S.add(a)
    return S

def _local_search(S, rng, deadline):
    # 1-flip: try adding each excluded vertex that has no conflict in S
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # candidate additions: vertices not in S with no conflict inside S
        for a in range(1, _N + 1):
            if a in S:
                continue
            if S & _CONFLICTS[a]:
                continue
            S.add(a)
            improved = True
        # 2-opt: remove one vertex x in S, then greedily add all now-possible
        for x in list(S):
            if time.time() > deadline:
                break
            S.remove(x)
            added = 0
            for a in range(1, _N + 1):
                if a in S or a == x:
                    continue
                if S & _CONFLICTS[a]:
                    continue
                S.add(a)
                added += 1
            if added >= 1:
                improved = True
            else:
                S.add(x)
    return S

def _hill_climb(S, rng, deadline):
    # random 1-swap: remove x, add y (y not in S, no conflict with S\{x})
    best = S
    best_size = len(S)
    for _ in range(4000):
        if time.time() > deadline:
            break
        x = rng.choice(sorted(S))
        S.remove(x)
        cands = [a for a in range(1, _N + 1) if a != x and a not in S and not (S & _CONFLICTS[a])]
        if not cands:
            S.add(x)
            continue
        y = rng.choice(cands)
        S.add(y)
        if len(S) > best_size:
            best = set(S)
            best_size = len(S)
        else:
            S.remove(y)
            S.add(x)
    return best

def construct():
    t0 = time.time()
    deadline = t0 + _T - _MARGIN
    rng = random.Random(12345)

    # multiple initial orderings
    orders = []
    # by degree ascending (sparse first)
    orders.append(sorted(range(1, _N + 1), key=lambda a: _DEG[a]))
    orders.append(sorted(range(1, _N + 1), key=lambda a: _DEG[a], reverse=True))
    orders.append(list(range(1, _N + 1)))
    orders.append(list(range(_N, 0, -1)))
    for _ in range(6):
        o = list(range(1, _N + 1))
        rng.shuffle(o)
        orders.append(o)

    best = set()
    for o in orders:
        if time.time() > deadline:
            break
        S = _greedy_init(o)
        S = _local_search(S, rng, deadline)
        if len(S) > len(best):
            best = S

    # hill-climb on the best
    best = _hill_climb(best, rng, deadline)

    # final polish: one more local-search pass
    best = _local_search(best, rng, deadline)

    result = sorted(best)
    # safety: if somehow invalid (should not be), fall back to odds
    if not _valid(set(result)):
        result = list(range(1, 257, 2))
    return result
