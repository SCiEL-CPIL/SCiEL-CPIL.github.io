# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

_DEADLINE = 620.0  # safety margin before 630s kill


def _pool_vectors(d):
    """Pool of low-weight integer vectors: weight 1 (e_i) and weight 2 (e_i - e_j, e_i + e_j, etc)."""
    vecs = []
    # weight 1: single +/-1
    for i in range(d):
        for s in (1, -1):
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            vecs.append(v)
    # weight 2: two non-zero coords, each in {-1,1}
    for i, j in itertools.combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vecs.append(v)
    return np.array(vecs, dtype=np.int64)


def _compatibility_matrix(pool):
    """
    Build a boolean matrix C where C[a,b] is True if points a and b can coexist
    under the rule: min pairwise dist^2 >= max norm^2.
    We compute this lazily; but for a static pool we precompute pairwise dist^2
    and norms.
    """
    n = pool.shape[0]
    # norms squared
    norms2 = np.einsum('ij,ij->i', pool, pool)
    # pairwise squared distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    G = pool @ pool.T
    dist2 = norms2[:, None] + norms2[None, :] - 2 * G
    return norms2, dist2


def _max_norm_of_set(pool, norms2, idx):
    return int(np.max(norms2[idx]))


def _min_pair_dist2(pool, dist2, idx):
    if len(idx) < 2:
        return np.inf
    sub = dist2[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), k=1)
    return int(np.min(sub[iu]))


def kissing_number11() -> np.ndarray:
    t0 = time.time()
    d = 11
    pool = _pool_vectors(d)
    norms2, dist2 = _compatibility_matrix(pool)
    n = pool.shape[0]

    # Greedy construction: repeatedly pick the vector that maximizes a score
    # (number of compatible unused partners) while maintaining validity.
    # We track the current set and its (max_norm2, min_pair_dist2).
    # A candidate c is admissible if:
    #   min_pair_dist2_with_existing >= max(max_norm2_existing, norm2_c)
    # We use a heuristic: build greedily, then local search.

    def feasible_with(cand, idx, norms2, dist2):
        nc = int(norms2[cand])
        if len(idx) == 0:
            return True
        cur_max = int(np.max(norms2[idx]))
        need = max(cur_max, nc)
        dsub = dist2[cand, idx]
        return bool(np.all(dsub >= need))

    def set_quality(idx, norms2, dist2):
        if len(idx) == 0:
            return 0
        if len(idx) == 1:
            return 1
        mn = int(np.min(dist2[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]))
        mx = int(np.max(norms2[idx]))
        return len(idx) if mn >= mx else -1

    # Greedy build
    used = np.zeros(n, dtype=bool)
    idx = []
    # sort pool by norm ascending to prefer low-norm (easier to satisfy)
    order = np.argsort(norms2)
    for cand in order:
        if used[cand]:
            continue
        if feasible_with(cand, idx, norms2, dist2):
            idx.append(cand)
            used[cand] = True

    best_idx = list(idx)
    best_score = len(best_idx)

    # Local search: try to add a random unused vector, and swap to improve.
    def try_add(idx, cand):
        return feasible_with(cand, idx, norms2, dist2)

    rng = np.random.default_rng(12345)
    remaining = [i for i in range(n) if not used[i]]

    # Iterative local improvement with time guard
    improved = True
    while improved and (time.time() - t0) < _DEADLINE:
        improved = False
        # Try adding each remaining candidate
        for cand in remaining:
            if try_add(best_idx, cand):
                best_idx.append(cand)
                remaining.remove(cand)
                improved = True
                break
        if improved:
            continue
        # Try swap: remove one from best, add a better one (net +1 not possible here,
        # so try replacing to allow future additions)
        # Random restarts of greedy with different ordering
        if (time.time() - t0) < _DEADLINE - 5:
            # Randomized greedy from scratch
            perm = rng.permutation(n)
            used2 = np.zeros(n, dtype=bool)
            idx2 = []
            for cand in perm:
                if feasible_with(cand, idx2, norms2, dist2):
                    idx2.append(cand)
                    used2[cand] = True
            if len(idx2) > best_score:
                best_idx = idx2
                best_score = len(idx2)
                remaining = [i for i in range(n) if not used2[i]]
                improved = True

    # Final validity check
    if len(best_idx) >= 2:
        mn = int(np.min(dist2[np.ix_(best_idx, best_idx)][np.triu_indices(len(best_idx), 1)]))
        mx = int(np.max(norms2[best_idx]))
        if mn < mx:
            # Fallback to the two trivial points
            best_idx = [0, n // 2]  # ensure two distinct
            # find two valid
            for a in range(n):
                for b in range(a + 1, n):
                    if int(dist2[a, b]) >= int(max(norms2[a], norms2[b])):
                        best_idx = [a, b]
                        break
                if len(best_idx) == 2:
                    break
    elif len(best_idx) == 1:
        best_idx = [0, 1] if n >= 2 else [0]

    result = np.around(pool[best_idx]).astype(np.int64)
    return result


# EVOLVE-BLOCK-END
