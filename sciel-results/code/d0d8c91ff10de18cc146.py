"""Base-B positional construction with digit-vector local search."""
import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 135.0          # hard wall-clock guard (limit is 150s)
_RNG = random.Random(20240517)


def _score_of(U):
    U = np.asarray(sorted(set(int(x) for x in U)), dtype=np.int64)
    if len(U) == 0 or U[0] != 0:
        return 0.0
    n = len(U)
    maxU = int(U[-1])
    if maxU <= 0:
        return 0.0
    S = U[:, None] + U[None, :]
    D = U[:, None] - U[None, :]
    ns = len(np.unique(S))
    nd = len(np.unique(D))
    denom = 2 * maxU + 1
    if ns <= 0 or nd <= 0:
        return 0.0
    return 1.0 + np.log(nd / ns) / np.log(denom)


def _expand(digits, B):
    k = len(digits)
    pos = np.empty(k, dtype=np.int64)
    pos[0] = 1
    for i in range(1, k):
        pos[i] = pos[i - 1] * B
    return (digits[:, None] * pos[None, :]).sum(axis=1)


def _eval(digits, B):
    U = _expand(digits, B)
    n = len(U)
    maxU = int(U.max())
    S = U[:, None] + U[None, :]
    D = U[:, None] - U[None, :]
    ns = len(np.unique(S))
    nd = len(np.unique(D))
    denom = 2 * maxU + 1
    if ns <= 0 or nd <= 0:
        return 0.0, U, ns, nd
    return 1.0 + np.log(nd / ns) / np.log(denom), U, ns, nd


def _local_search(digits, B, time_limit, rng):
    digits = digits.copy()
    best, U, ns, nd = _eval(digits, B)
    best_digits = digits.copy()
    best_U = U.copy()
    k = len(digits)
    while time.time() < time_limit:
        if k == 0:
            break
        i = rng.randrange(k)
        ndig = digits.copy()
        ndig[i] = 1 - ndig[i]
        if ndig[0] == 0:          # must keep 0 in the set
            continue
        s, U, ns, nd = _eval(ndig, B)
        if s > best + 1e-15:
            best = s
            digits = ndig
            best_digits = digits.copy()
            best_U = U.copy()
            k = len(digits)
    return best, best_digits, best_U


def construct():
    t_end = _DEADLINE
    rng = _RNG
    best_score = 0.0
    best_set = [0]

    def record(score, U):
        nonlocal best_score, best_set
        if score > best_score + 1e-15:
            best_score = score
            best_U = np.sort(np.unique(np.asarray(U, dtype=np.int64)))
            best_set = [int(x) for x in best_U]

    # ---- seed: base-3 with digits {0,1} ----
    base_seeds = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1,
         1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    ]
    for B in (3, 4):
        for seed in base_seeds:
            s, U, _, _ = _eval(np.array(seed, dtype=np.int64), B)
            record(s, U)

    # ---- local search over the digit vector, larger bases later ----
    for B in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        if time.time() >= t_end:
            break
        remaining = t_end - time.time()
        if remaining < 4.0:
            break
        # random start digit vector (leading digit = 1 to keep max large)
        digits = np.zeros(2, dtype=np.int64)
        digits[0] = 1
        digits[1] = 1
        s, U, _, _ = _eval(digits, B)
        record(s, U)
        s, digits, U = _local_search(digits, B, t_end - 1.0, rng)
        record(s, U)

        # second randomized start
        if time.time() >= t_end:
            break
        L = rng.randrange(3, 7)
        d2 = np.zeros(L, dtype=np.int64)
        d2[0] = 1
        for i in range(1, L):
            d2[i] = 1 if rng.random() < 0.5 else 0
        if d2[0] == 0:
            d2[0] = 1
        s2, U2, _, _ = _eval(d2, B)
        record(s2, U2)
        s2, d2, U2 = _local_search(d2, B, t_end - 1.0, rng)
        record(s2, U2)

    # ---- final exact verification of the best set ----
    if best_set:
        final = _score_of(best_set)
        if final > 0:
            best_score = max(best_score, final)
    return best_set
