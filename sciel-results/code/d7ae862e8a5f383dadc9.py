"""Kissing number in dimension 11.

Strategy (combined):
  1. Seed with the full D11 weight-2 set (220 vectors, ±1 on a 2-subset).
     This is a proven valid configuration and gives a strong floor.
  2. Greedily add weight-4 binary vectors (±1 on a 4-subset). The conflict graph
     is sparse: two weight-4 vectors with intersection <= 2 are always compatible
     (|dot| <= 2), so only supports sharing 3 indices need checking. We build that
     graph once, then process supports in a heuristic order (degree ascending,
     with randomization) and, for each support, vectorize-select the maximal
     subset of its 16 sign patterns compatible with all already-selected vectors.
  3. Several randomized restarts; keep the largest valid set; hard deadline guard.
"""
import numpy as np
import time
import itertools
from collections import defaultdict


def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # safety margin

    D = 11

    # ---- 1. D11 weight-2 seed (220 vectors) ----
    w2 = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    w2.append(v)
    w2_arr = np.array(w2, dtype=np.int64)
    w2_norm_sq = 2
    w2_dot = w2_arr @ w2_arr.T
    # valid pairs: distinct (i != j) and dist_sq = 4 - 2*dot >= 2  ->  dot <= 1
    w2_ok = (w2_dot <= 1)
    np.fill_diagonal(w2_ok, False)

    # ---- 2. Support graph for weight-4 vectors ----
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330

    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)

    adj = [set() for _ in range(n_sup)]
    for tets in tri_to_tet.values():
        L = len(tets)
        for a in range(L):
            for b in range(a + 1, L):
                adj[tets[a]].add(tets[b])
                adj[tets[b]].add(tets[a])

    # precompute, per support, the list of the 16 sign patterns as (16,4) int8
    sign_vecs4 = []
    for s in supports:
        rows = []
        for mask in range(16):
            row = np.zeros(D, dtype=np.int8)
            for k in range(4):
                row[s[k]] = 1 if (mask >> k) & 1 else -1
            rows.append(row)
        sign_vecs4.append(np.array(rows, dtype=np.int8))  # (16, D)

    # neighbor lists as arrays (for fast indexing)
    adj_arr = [np.array(sorted(a), dtype=np.int64) for a in adj]

    def build_set(order, seed):
        """Greedy over supports in `order`; returns (array, count)."""
        selected = [np.empty(0, dtype=np.int64) for _ in range(n_sup)]
        processed = [False] * n_sup

        for idx in order:
            cand = sign_vecs4[idx]  # (16, D)
            keep = np.ones(16, dtype=bool)

            # conflicts with weight-2 seed (dot <= 1)
            d2 = cand @ w2_arr.T  # (16, 220)
            keep &= (d2 <= 1).all(axis=1)

            # conflicts with already-selected weight-4 from processed neighbors
            for nb in adj_arr[idx]:
                if not processed[nb]:
                    continue
                sel = selected[nb]
                if sel.size == 0:
                    continue
                d4 = cand @ sel.T  # (16, m)
                keep &= (d4 <= 1).all(axis=1)

            chosen = np.nonzero(keep)[0]
            if chosen.size > 0:
                selected[idx] = cand[chosen]

            processed[idx] = True

        parts = [w2_arr]
        for idx in range(n_sup):
            if selected[idx].size:
                parts.append(selected[idx])
        return np.vstack(parts)

    # ---- 3. Search: randomized restarts, keep best ----
    best = w2_arr.copy()
    best_count = len(w2_arr)

    rng = np.random.default_rng(12345)
    max_restarts = 60

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            order = np.array(sorted(range(n_sup), key=lambda i: len(adj[i])), dtype=np.int64)
        else:
            # mix degree heuristic with randomness
            deg = np.array([len(adj[i]) for i in range(n_sup)], dtype=np.float64)
            keys = deg + rng.random(n_sup) * 2.0
            order = np.argsort(keys).astype(np.int64)

        arr = build_set(order, restart)
        if arr.shape[0] > best_count:
            best = arr
            best_count = arr.shape[0]

    return best


if __name__ == "__main__":
    res = construct()
    print("shape:", res.shape)
    # quick self-check
    a = res @ res.T
    n = np.einsum("ij,ij->i", res, res)
    dist = n[:, None] + n[None, :] - 2 * a
    np.fill_diagonal(dist, 1e18)
    mn = dist.min()
    mx = n.max()
    print("min dist^2:", mn, "max norm^2:", mx, "ok:", mn >= mx)
