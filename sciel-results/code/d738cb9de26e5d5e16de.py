"""Construct a finite set U of nonnegative integers (containing 0) to maximize
the score 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)."""
import time
import math
import random

_START = time.time()
_DEADLINE = 112.0  # seconds; leave margin under the 120 s wall-clock kill


def _score_of(U):
    if not U or 0 not in U:
        return -1.0
    mx = max(U)
    if mx <= 0:
        return 1.0
    s = set()
    d = set()
    for a in U:
        for b in U:
            s.add(a + b)
            d.add(a - b)
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 1.0
    ratio = len(d) / len(s)
    return 1.0 + math.log(ratio) / log_den


def _geometric(b, k):
    return [b ** i for i in range(k + 1)]


def _sa_search(N, iters, seed):
    """Simulated annealing over subsets of [0..N] that always contain 0."""
    rng = random.Random(seed)
    # start with a geometric-ish or random set
    bits = [False] * (N + 1)
    bits[0] = True
    for i in range(1, N + 1):
        bits[i] = rng.random() < 0.5
    cur = [i for i in range(N + 1) if bits[i]]
    cur_score = _score_of(cur)
    best = list(cur)
    best_score = cur_score
    T = 0.5
    for it in range(iters):
        if time.time() - _START > _DEADLINE:
            break
        # propose a flip (never flip index 0)
        i = rng.randint(1, N)
        bits[i] = not bits[i]
        new = [j for j in range(N + 1) if bits[j]]
        ns = _score_of(new)
        if ns >= cur_score or rng.random() < math.exp((ns - cur_score) / max(T, 1e-9)):
            cur = new
            cur_score = ns
            if ns > best_score:
                best_score = ns
                best = list(new)
        else:
            bits[i] = not bits[i]  # revert
        T *= 0.999
    return best, best_score


def construct():
    best_U = [0, 1, 3]
    best_score = _score_of(best_U)

    def consider(U):
        nonlocal best_U, best_score
        if not U or 0 not in U:
            return
        sc = _score_of(U)
        if sc > best_score:
            best_score = sc
            best_U = list(U)

    # Family: geometric progressions
    for k in range(1, 20):
        for b in range(2, 40):
            if time.time() - _START > _DEADLINE:
                break
            consider(_geometric(b, k))
        if time.time() - _START > _DEADLINE:
            break

    # Family: simulated annealing over small ranges
    for N in [6, 8, 10, 12, 15, 18, 22]:
        if time.time() - _START > _DEADLINE:
            break
        # budget per N
        budget = 4000
        for seed in range(3):
            if time.time() - _START > _DEADLINE:
                break
            U, sc = _sa_search(N, budget, seed + N)
            consider(U)

    return best_U
