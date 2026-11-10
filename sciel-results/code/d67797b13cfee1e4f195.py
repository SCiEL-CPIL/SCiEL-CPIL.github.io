"""Rewritten: vectorized maximality, redundancy pruning, randomized restarts."""
import time
import itertools
import numpy as np

N = 16
ALL_POINTS = np.array(list(itertools.product(range(N), repeat=2)), dtype=np.int64)
M = N * N
DEADLINE = 320.0


def _dist2_to_set(A, B):
    d = A[:, None, :] - B[None, :, :]
    return np.sum(d * d, axis=2)


def _addable_mask(S):
    k = len(S)
    if k < 2:
        addable = np.ones(M, dtype=bool)
        if k >= 1:
            addable[S[:, 0] * N + S[:, 1]] = False
        return addable
    D_ps = _dist2_to_set(ALL_POINTS, S)  # (M, k)
    Ds = np.sort(D_ps, axis=1)
    cond1 = np.any(Ds[:, 1:] == Ds[:, :-1], axis=1)
    D_ss = _dist2_to_set(S, S)  # (k, k)
    bad2 = np.zeros(M, dtype=bool)
    for a in range(k):
        forb = np.unique(D_ss[a, np.arange(k) != a])
        bad2 |= np.isin(D_ps[:, a], forb)
    addable = ~(cond1 | bad2)
    addable[S[:, 0] * N + S[:, 1]] = False
    return addable


def _is_maximal(S):
    if len(S) < 2:
        return len(S) >= M
    return not np.any(_addable_mask(S))


def _essential_mask(S):
    k = len(S)
    if k == 0:
        return np.zeros(0, dtype=bool)
    ess = np.zeros(k, dtype=bool)
    for i in range(k):
        if not _is_maximal(np.delete(S, i, axis=0)):
            ess[i] = True
    return ess


def _prune_redundant(S):
    S = S.copy()
    while len(S) > 0:
        ess = _essential_mask(S)
        if ess.all():
            break
        S = S[ess]
    return S


def _greedy(order):
    chosen = []
    for idx in order:
        p = ALL_POINTS[idx]
        S = np.array(chosen, dtype=np.int64) if chosen else np.empty((0, 2), dtype=np.int64)
        if _addable_mask(S)[idx]:
            chosen.append(p)
    return np.array(chosen, dtype=np.int64) if chosen else np.empty((0, 2), dtype=np.int64)


def construct():
    start = time.time()
    best = None
    best_len = M + 1

    # Deterministic row-major baseline + prune
    try:
        S = _greedy(np.arange(M))
        S = _prune_redundant(S)
        if len(S) < best_len:
            best_len = len(S)
            best = S.copy()
    except Exception:
        pass

    rng = np.random.default_rng(777)
    while time.time() - start < DEADLINE:
        try:
            S = _greedy(rng.permutation(M))
            S = _prune_redundant(S)
            if len(S) < best_len:
                best_len = len(S)
                best = S.copy()
        except Exception:
            continue

    if best is None:
        return [tuple(p) for p in ALL_POINTS]
    return [tuple(p) for p in best]
