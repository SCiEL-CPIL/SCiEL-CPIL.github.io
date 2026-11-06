"""Rewritten: randomized greedy build + maximality-preserving pruning,
with deadline guard and multi-restart. Minimize |set| of a maximal
no-isosceles set on the [0,16)^2 grid."""

import time
import numpy as np

N = 16
M = N * N  # 256 grid points
DEADLINE = 300.0
START = time.time()


def _time_left():
    return DEADLINE - (time.time() - START)


def _init_dist():
    xs = np.arange(N)
    ys = np.arange(N)
    X, Y = np.meshgrid(xs, ys, indexing="ij")  # [x][y]
    P = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.int64)  # (256,2)
    D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)  # (256,256) int64
    return P, D


_P, _D = _init_dist()
_D = _D.astype(np.int64)


def _blocked_mask(D, S_idx, cand):
    """Return bool array: for each candidate point, is it blocked by set S_idx?
    S_idx: 1D array of indices in set. cand: 1D array of candidate indices.
    A candidate p is blocked if exists a,b in S with {p,a,b} isosceles."""
    if len(S_idx) < 2:
        return np.zeros(len(cand), dtype=bool)
    DS = D[np.ix_(S_idx, S_idx)]  # (k,k)
    out = np.zeros(len(cand), dtype=bool)
    for i, p in enumerate(cand):
        dp = D[p, S_idx]  # (k,)
        # pair-wise: (k,k) booleans
        E = (dp[:, None] == dp[None, :]) | (dp[:, None] == DS) | (dp[None, :] == DS)
        out[i] = E.any()
    return out


def _is_maximal(D, S_idx):
    """True if no point outside S_idx can be added (set is maximal)."""
    mask = np.zeros(M, dtype=bool)
    mask[S_idx] = True
    cand = np.where(~mask)[0]
    if len(cand) == 0:
        return True
    return bool(_blocked_mask(D, S_idx, cand).all())


def _build(D, order):
    """Greedy add in given order; return list of indices."""
    S = []
    S_set = set()
    for p in order:
        if p in S_set:
            continue
        if len(S) < 2:
            S.append(p)
            S_set.add(p)
            continue
        # check not isosceles with any pair
        dp = D[p, S]
        DS = D[np.ix_(S, S)]
        E = (dp[:, None] == dp[None, :]) | (dp[:, None] == DS) | (dp[None, :] == DS)
        if not E.any():
            S.append(p)
            S_set.add(p)
    return S


def _prune(D, S):
    """Remove points whose removal keeps the set maximal. Repeat until stable."""
    S = list(S)
    changed = True
    while changed and _time_left() > 5.0:
        changed = False
        for i in range(len(S)):
            trial = S[:i] + S[i + 1:]
            if len(trial) < 2:
                continue
            if _is_maximal(D, trial):
                S = trial
                changed = True
                break  # restart scan after a change
    return S


def construct():
    rng = np.random.default_rng(12345)
    all_idx = np.arange(M)
    best = None
    while _time_left() > 3.0:
        order = rng.permutation(all_idx)
        S = _build(_D, order)
        S = _prune(_D, S)
        if best is None or len(S) < len(best):
            best = S
            if len(best) <= 12:  # good enough early stop
                break
    if best is None:
        # fallback: deterministic row-major greedy
        S = _build(_D, all_idx)
        S = _prune(_D, S)
        best = S
    # convert indices to (x,y)
    pts = [tuple(_P[i].tolist()) for i in best]
    return pts
