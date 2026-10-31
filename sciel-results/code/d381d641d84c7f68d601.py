import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # ---- Candidate pool: all norm^2 = 4 vectors, so max_norm^2 = 4 ----
    # Type A: weight-4 in {-1,0,1}^11
    vA = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            vA.append(v)
    # Type B: weight-2 in {-2,0,2}^11
    vB = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = 2 * s
            vB.append(v)

    V = np.array(vA + vB, dtype=np.int64)
    N = len(V)
    if N == 0:
        return np.zeros((0, D), dtype=np.int64)

    # ---- Conflict graph: conflict iff dot > 2 (i.e. dist^2 < 4) ----
    dots = V @ V.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    conflicts = conflicts.astype(bool)
    degrees = conflicts.sum(axis=1).astype(np.int64)
    arange = np.arange(N)

    best_selected = None
    best_count = 0

    # ---- Strategy 1: greedy minimum-degree MIS with random restarts ----
    for restart in range(40):
        if time.time() > deadline - 40:
            break
        rng = np.random.default_rng(1000 + restart * 13)
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        deg = degrees.copy()
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            alive_deg = deg[alive_idx]
            min_deg = np.min(alive_deg)
            cands = alive_idx[alive_deg == min_deg]
            chosen = cands[rng.integers(len(cands))]
            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (arange == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                deg[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        cnt = selected.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = selected.copy()

    # ---- Local search: add free, 1-opt (remove1/add1), 2-opt (remove2/add2) ----
    if best_selected is not None:
        sel = best_selected.copy()
        rng_ls = np.random.default_rng(77)

        # Add all free vertices (no conflict with selected)
        def add_free(s):
            changed = True
            while changed:
                if time.time() > deadline - 20:
                    return s
                changed = False
                non_sel = np.where(~s)[0]
                if len(non_sel) == 0:
                    break
                ok = ~np.any(conflicts[non_sel] & s[:, None], axis=1)
                add_idx = non_sel[ok]
                if len(add_idx) > 0:
                    s[add_idx] = True
                    changed = True
            return s

        sel = add_free(sel)
        if sel.sum() > best_count:
            best_count = sel.sum()
            best_selected = sel.copy()

        # 1-opt: remove one selected u, add one non-selected v (v only conflicts with u)
        improved = True
        while improved:
            if time.time() > deadline - 20:
                break
            improved = False
            sel_idx = np.where(sel)[0]
            non_sel_idx = np.where(~sel)[0]
            if len(non_sel_idx) == 0:
                break
            # For each non-selected v, its selected-neighbors
            neigh = conflicts[non_sel_idx] & sel[:, None]
            counts = neigh.sum(axis=1)
            cand = non_sel_idx[counts == 1]
            if len(cand) == 0:
                break
            # try each candidate; accept first that yields net gain (always >=0 here)
            for v in cand:
                u = neigh[non_sel_idx == v][0][0]
                sel[u] = False
                sel[v] = True
                sel = add_free(sel)
                if sel.sum() > best_count:
                    best_count = sel.sum()
                    best_selected = sel.copy()
                improved = True
                break

        # 2-opt: remove two selected {u,w}, add two non-selected {v,z}
        improved = True
        iters = 0
        while improved and iters < 200:
            if time.time() > deadline - 15:
                break
            improved = False
            iters += 1
            sel_idx = np.where(sel)[0]
            non_sel_idx = np.where(~sel)[0]
            if len(sel_idx) < 2 or len(non_sel_idx) < 2:
                break
            best_gain = 0
            best_op = None
            # sample pairs for speed
            n_trials = 400
            for _ in range(n_trials):
                if time.time() > deadline - 15:
                    break
                v, z = rng_ls.choice(non_sel_idx, size=2, replace=False)
                if conflicts[v, z]:
                    continue
                sel_neigh = np.unique(np.where((conflicts[v] | conflicts[z]) & sel)[0])
                if len(sel_neigh) > 2:
                    continue
                if len(sel_neigh) == 2:
                    u, w = sel_neigh
                elif len(sel_neigh) == 1:
                    u = sel_neigh[0]
                    w = None
                else:
                    continue
                # tentative
                if w is not None:
                    sel[u] = False
                    sel[w] = False
                else:
                    sel[u] = False
                sel[v] = True
                sel[z] = True
                sel = add_free(sel)
                gain = sel.sum() - best_count  # relative
                if gain > best_gain:
                    best_gain = gain
                    best_op = (u, w, v, z)
                # revert to pre-op state
                # easiest: rebuild from best_selected? Instead, just track and re-derive.
                # We'll revert by undoing: remove v,z, re-add u(,w)
                sel[v] = False
                sel[z] = False
                sel[u] = True
                if w is not None:
                    sel[w] = True
                sel = add_free(sel)
            if best_op is not None and best_gain > 0:
                u, w, v, z = best_op
                if w is not None:
                    sel[u] = False
                    sel[w] = False
                else:
                    sel[u] = False
                sel[v] = True
                sel[z] = True
                sel = add_free(sel)
                if sel.sum() > best_count:
                    best_count = sel.sum()
                    best_selected = sel.copy()
                improved = True

        if sel.sum() > best_count:
            best_count = sel.sum()
            best_selected = sel.copy()

    # ---- Final: add free + verify constraint, drop violators ----
    result = V[best_selected]
    result = add_free(best_selected.copy())
    result = V[result]

    # Verify: min pairwise dist^2 >= 4 (max norm^2 = 4)
    n = len(result)
    if n > 1:
        norms_sq = np.sum(result.astype(np.float64) ** 2, axis=1)
        max_norm_sq = np.max(norms_sq)
        valid = np.ones(n, dtype=bool)
        batch = 500
        for i in range(0, n, batch):
            if time.time() > deadline - 2:
                break
            B = result[i:i + batch].astype(np.float64)
            dots_b = B @ result.T
            ns = np.sum(B ** 2, axis=1)
            allns = np.sum(result.astype(np.float64) ** 2, axis=1)
            dist_sq = ns[:, None] + allns[None, :] - 2 * dots_b
            for r in range(len(B)):
                gi = i + r
                if not valid[gi]:
                    continue
                d = dist_sq[r, :].copy()
                d[gi] = np.inf
                bad = np.where((d < max_norm_sq - 1e-6) & valid)[0]
                if len(bad) > 0:
                    valid[gi] = False
                    valid[bad] = False
        result = result[valid]

    return result
