"""Covers K_q(n,1): vectorized greedy seed + delete/swap local search.
Fresh design; explicit wall-clock deadline guard with safety margin."""
import time
import numpy as np

def _tuples_to_flat(tups):
    return np.array(tups, dtype=np.int64).reshape(-1, n)

def _all_neighbors(flat, q, n):
    m = flat.shape[0]
    if m == 0:
        return np.zeros((0, n), dtype=np.int64)
    parts = [flat]
    for pos in range(n):
        for v in range(q):
            nf = flat.copy()
            nf[:, pos] = v
            mask = (nf[:, pos] != flat[:, pos])
            if mask.any():
                parts.append(nf[mask])
    return np.concatenate(parts, axis=0)

def _is_cover(flat, q, n, radius):
    if radius == 1:
        N = q ** n
        allw = np.unravel_index(np.arange(N, dtype=np.int64), (q,) * n)
        allw = np.stack(allw, axis=1).astype(np.int64)
        neigh = _all_neighbors(flat, q, n)
        return np.intersect1d(allw, neigh, assume_unique=True).shape[0] == N
    covered = np.zeros(q ** n, dtype=bool)
    cur = flat
    for _ in range(radius):
        cur = _all_neighbors(cur, q, n)
    idxs = np.ravel_multi_index(tuple(cur.T), (q,) * n)
    covered[idxs] = True
    return covered.all()

def _greedy(q, n, radius, deadline):
    N = q ** n
    allw = np.unravel_index(np.arange(N, dtype=np.int64), (q,) * n)
    allw = np.stack(allw, axis=1).astype(np.int64)
    neigh = _all_neighbors(allw, q, n)
    neigh_idx = np.ravel_multi_index(tuple(neigh.T), (q,) * n)
    neigh_of_word = [np.where(neigh_idx == w)[0] for w in range(N)]
    covered = np.zeros(N, dtype=bool)
    chosen = []
    for it in range(N):
        if time.time() > deadline:
            break
        if covered.all():
            break
        cand = np.where(~covered)[0]
        if cand.size == 0:
            break
        gains = np.array([covered[neigh_of_word[w]].sum() for w in cand], dtype=np.int64)
        best_local = cand[np.argmax(gains)]
        chosen.append(best_local)
        covered[neigh_of_word[best_local]] = True
    return np.array(chosen, dtype=np.int64).reshape(-1, n)

def construct(q, n, radius):
    deadline = time.time() + 110.0
    N = q ** n
    flat = _greedy(q, n, radius, deadline)
    if flat.shape[0] == 0:
        flat = np.zeros((1, n), dtype=np.int64)

    allw = np.unravel_index(np.arange(N, dtype=np.int64), (q,) * n)
    allw = np.stack(allw, axis=1).astype(np.int64)
    neigh = _all_neighbors(allw, q, n)
    neigh_idx = np.ravel_multi_index(tuple(neigh.T), (q,) * n)
    neigh_of_word = [np.where(neigh_idx == w)[0] for w in range(N)]

    def recompute(flat_arr):
        c = np.zeros(N, dtype=bool)
        for w in flat_arr:
            c[neigh_of_word[w[0]]] = True
        return c

    # ---- delete pass (repeated until stable) ----
    changed = True
    while changed and time.time() < deadline:
        changed = False
        covered = recompute(flat)
        m = flat.shape[0]
        # coverage count per word
        cnt = np.zeros(N, dtype=np.int64)
        for w in flat:
            cnt[neigh_of_word[w[0]]] += 1
        for i in range(m):
            if time.time() > deadline:
                break
            wi = flat[i, 0]
            # words covered only by i
            only = (cnt[neigh_of_word[wi]] == 1)
            if not only.any():
                flat = np.delete(flat, i, axis=0)
                changed = True
                break

    # ---- swap pass: try removing i, adding best j that covers the only-i words ----
    improved = True
    while improved and time.time() < deadline:
        improved = False
        covered = recompute(flat)
        cnt = np.zeros(N, dtype=np.int64)
        for w in flat:
            cnt[neigh_of_word[w[0]]] += 1
        m = flat.shape[0]
        for i in range(m):
            if time.time() > deadline:
                break
            wi = flat[i, 0]
            only = (cnt[neigh_of_word[wi]] == 1)
            if not only.any():
                continue  # deletable, handled by delete pass
            # words that would be uncovered
            uncov = only.nonzero()[0]
            # find j maximizing coverage of uncov
            # score_j = |neigh_of_word[j] ∩ uncov|
            best_j = -1
            best_score = len(uncov)  # need to cover ALL to keep size same
            for j in range(N):
                if j == wi:
                    continue
                s = covered[neigh_of_word[j]].sum()  # not right metric
                # metric: how many of uncov are covered by j
                s = np.isin(neigh_of_word[j], uncov).sum()
                if s > best_score:
                    best_score = s
                    best_j = j
            if best_j >= 0 and best_score == len(uncov):
                flat = np.delete(flat, i, axis=0)
                flat = np.vstack([flat, allw[best_j]])
                improved = True
                break

    return [list(map(int, row)) for row in flat]
