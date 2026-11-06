"""Solver for equal-lcm-triple-free subsets of [1,128]."""
# EVOLVE-BLOCK-START
import time
import math

_TIME_LIMIT = 85.0
_start = time.time()

def _elapsed():
    return time.time() - _start

def _lcm(a, b):
    return a // math.gcd(a, b) * b

def _can_add(S, x, lcm_cache):
    """Return True if x can be added to S without forming a bad triple.
    A bad triple with x exists iff there are a,b in S with lcm(a,b)==lcm(a,x)==lcm(b,x)."""
    for i in range(len(S)):
        a = S[i]
        lab = lcm_cache[(a, x)]
        for j in range(i + 1, len(S)):
            b = S[j]
            if lcm_cache[(a, b)] == lab and lcm_cache[(b, x)] == lab:
                return False
    return True

def _greedy(order, lcm_cache):
    S = []
    for x in order:
        if _can_add(S, x, lcm_cache):
            S.append(x)
    return S

def _local_search(S, lcm_cache, max_iters=100000):
    S = list(S)
    best_len = len(S)
    for it in range(max_iters):
        if _elapsed() > _TIME_LIMIT - 3:
            break
        improved = False
        # adds
        for x in range(1, 129):
            if x not in S and _can_add(S, x, lcm_cache):
                S.append(x)
                best_len = len(S)
                improved = True
                break
        if improved:
            continue
        # swaps
        for i in range(len(S)):
            removed = S[i]
            for x in range(1, 129):
                if x == removed or x in S:
                    continue
                trial = S[:i] + S[i+1:] + [x]
                if _can_add(trial, x, lcm_cache) or True:
                    # recheck full: _can_add on trial without x
                    rest = trial[:-1]
                    if _can_add(rest, x, lcm_cache):
                        S = trial
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return S

def _full_valid(S, lcm_cache):
    S = list(S)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            lab = lcm_cache[(a, b)]
            for k in range(j + 1, len(S)):
                c = S[k]
                if lcm_cache[(a, c)] == lab and lcm_cache[(b, c)] == lab:
                    return False
    return True

def construct():
    nonlocal _start
    _start = time.time()

    # Precompute lcm cache for all pairs in 1..128
    lcm_cache = {}
    for a in range(1, 129):
        for b in range(1, 129):
            lcm_cache[(a, b)] = _lcm(a, b)

    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        [x for x in range(1, 129) if x % 2 == 1],
        [x for x in range(1, 129) if x % 2 == 0],
        sorted(range(1, 129), key=lambda x: (x.bit_count(), x)),
        sorted(range(1, 129), key=lambda x: (x.bit_count(), -x)),
        [x for x in range(1, 129) if x % 3 != 0],
        [x for x in range(1, 129) if x % 3 == 0],
        [x for x in range(1, 129) if (x % 2 == 1) or (x % 3 != 0)],
        [x for x in range(1, 129) if (x % 2 == 0) or (x % 3 != 0)],
        sorted(range(1, 129), key=lambda x: -x.bit_count()),
        sorted(range(1, 129), key=lambda x: x.bit_count()),
        [x for x in range(1, 129) if x % 5 != 0],
        [x for x in range(1, 129) if x % 4 == 0],
    ]

    best = []
    best_len = 0
    for order in orders:
        if _elapsed() > _TIME_LIMIT - 15:
            break
        S = _greedy(order, lcm_cache)
        if len(S) > best_len:
            best = S
            best_len = len(S)

    best = _local_search(best, lcm_cache, max_iters=100000)
    best_len = len(best)

    if _elapsed() < _TIME_LIMIT - 10:
        for order in orders[:8]:
            S = _greedy(order, lcm_cache)
            S = _local_search(S, lcm_cache, max_iters=30000)
            if len(S) > best_len:
                best = S
                best_len = len(S)
            if _elapsed() > _TIME_LIMIT - 5:
                break

    if not best:
        best = [1]
    if not _full_valid(best, lcm_cache):
        # rebuild safely
        best = _greedy(list(range(1, 129)), lcm_cache)
        if not _full_valid(best, lcm_cache):
            best = [1]
    return best
# EVOLVE-BLOCK-END
