"""Rewritten solver: minimum maximal no-isosceles set on 16x16 grid.

Pipeline: vectorized feasibility, center-out greedy seed, 2-opt local search,
multi-restart, deadline guard.
"""
import time
import numpy as np
import itertools

N = 16
DEADLINE = time.time() + 300.0  # evaluator kills at 330s; keep 30s safety margin
BEST = [None]  # best (smallest) maximal set found so far


def _dist2(a, b, c):
    """Squared distances of the triangle (a,b,c). a,b,c are (x,y) tuples."""
    d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    return d1, d2, d3


def _is_isosceles(a, b, c):
    d1, d2, d3 = _dist2(a, b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _feasible_vec(new_pt, pts):
    """Vectorized check: can new_pt be added to pts without creating isosceles?

    new_pt: (x,y) tuple
    pts: list of (x,y) tuples
    Returns True if adding new_pt keeps the set no-isosceles.
    """
    if len(pts) < 2:
        return True
    new_pt = np.asarray(new_pt, dtype=np.int64)
    P = np.asarray(pts, dtype=np.int64)  # (m, 2)
    # distances from new_pt to each point
    diff_new = P - new_pt  # (m, 2)
    d_new = (diff_new ** 2).sum(axis=1)  # (m,)
    # pairwise distances among existing points: (m, m)
    diff = P[:, None, :] - P[None, :, :]  # (m, m, 2)
    d_pair = (diff ** 2).sum(axis=2)  # (m, m)
    # For each pair (i, j) in existing points, adding new_pt creates triangle
    # (new_pt, P_i, P_j) with sides d_new[i], d_new[j], d_pair[i,j].
    # Isosceles iff any two of the three are equal.
    # d_new[i] == d_new[j]  ->  same distance from new_pt to two existing pts
    # d_new[i] == d_pair[i,j]
    # d_new[j] == d_pair[i,j]
    # We only need to check pairs i < j.
    iu = np.triu_indices(len(pts), k=1)
    dn_i = d_new[iu[0]]
    dn_j = d_new[iu[1]]
    dp = d_pair[iu[0], iu[1]]
    bad = (dn_i == dn_j) | (dn_i == dp) | (dn_j == dp)
    return not np.any(bad)


def _is_maximal(pts):
    """Check that no grid point can be added (maximality)."""
    if len(pts) == 0:
        return True
    P = np.asarray(pts, dtype=np.int64)
    m = len(pts)
    # Precompute pairwise distances among existing points
    diff = P[:, None, :] - P[None, :, :]
    d_pair = (diff ** 2).sum(axis=2)  # (m, m)
    iu = np.triu_indices(m, k=1)
    dp_mat = d_pair[iu[0], iu[1]]  # (num_pairs,)
    for x in range(N):
        for y in range(N):
            cand = (x, y)
            if cand in pts:
                continue
            cand_arr = np.asarray(cand, dtype=np.int64)
            d_new = ((P - cand_arr) ** 2).sum(axis=1)  # (m,)
            dn_i = d_new[iu[0]]
            dn_j = d_new[iu[1]]
            bad = (dn_i == dn_j) | (dn_i == dp_mat) | (dn_j == dp_mat)
            if not np.any(bad):
                return False  # cand can be added -> not maximal
    return True


def _greedy(order):
    """Greedy construction following the given point order."""
    chosen = []
    for p in order:
        if _feasible_vec(p, chosen):
            chosen.append(p)
    return chosen


def _local_search(pts, max_iter=200):
    """2-opt local search: try swaps (remove one, add another) and deletes.

    Returns a (possibly smaller) maximal no-isosceles set.
    """
    pts = list(pts)
    improved = True
    iters = 0
    while improved and iters < max_iter and time.time() < DEADLINE - 5:
        improved = False
        iters += 1
        n = len(pts)
        if n == 0:
            break
        # Precompute for feasibility checks
        P = np.asarray(pts, dtype=np.int64)
        m = n
        diff = P[:, None, :] - P[None, :, :]
        d_pair = (diff ** 2).sum(axis=2)
        iu = np.triu_indices(m, k=1)
        dp_mat = d_pair[iu[0], iu[1]]

        # Build set of occupied points for fast lookup
        occ = set(pts)

        # Try deletions: remove a point, check if still maximal; if yes and size decreases, keep
        # Actually we want to REDUCE size while staying maximal.
        # A deletion keeps the set no-isosceles (removing points can't create isosceles).
        # But it may break maximality. We want the smallest maximal set.
        # So: try removing a point; if the result is still maximal, accept (size -1).
        # Also try swaps: remove p_i, add q (not in set), if result is no-isosceles and maximal.

        # First pass: try all single deletions
        for idx in range(n):
            if time.time() > DEADLINE - 5:
                break
            trial = pts[:idx] + pts[idx + 1:]
            if _is_maximal(trial):
                pts = trial
                improved = True
                break  # restart scan

        if improved:
            continue

        # Second pass: try swaps (remove one, add one) to possibly reduce size is not possible
        # via swap (size same). Swaps help escape local minima in maximality but not size.
        # For size reduction, deletions are the key. But sometimes a swap enables a later deletion.
        # Try a limited number of swaps to diversify.
        all_pts = list(itertools.product(range(N), repeat=2))
        # Try adding a point then removing a different one (net size same) to change structure
        # This is expensive; do a limited random subset.
        if n > 0 and iters < max_iter // 2:
            # Try a few random swaps
            for _ in range(5):
                if time.time() > DEADLINE - 5:
                    break
                idx = np.random.randint(0, len(pts))
                remove_pt = pts[idx]
                # Pick a random candidate to add
                cand = tuple(np.random.randint(0, N, size=2))
                if cand in occ or cand == remove_pt:
                    continue
                trial = [p for i, p in enumerate(pts) if i != idx]
                # Check feasibility of adding cand to trial
                if _feasible_vec(cand, trial):
                    trial.append(cand)
                    # Check maximality
                    if _is_maximal(trial):
                        pts = trial
                        improved = True
                        break
    return pts


def _center_out_order():
    """Order points by distance to grid center, center-out."""
    cx, cy = (N - 1) / 2.0, (N - 1) / 2.0
    pts = list(itertools.product(range(N), repeat=2))
    pts.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    return pts


def _random_order(rng):
    pts = list(itertools.product(range(N), repeat=2))
    rng.shuffle(pts)
    return pts


def _row_major_order():
    return list(itertools.product(range(N), repeat=2))


def construct():
    """Return a maximal no-isosceles set of grid points (list of (x,y))."""
    best_set = None
    best_size = N * N + 1
    rng = np.random.default_rng(12345)

    # Restart 0: center-out greedy (usually good)
    order0 = _center_out_order()
    t0 = time.time()
    seed0 = _greedy(order0)
    # Make it maximal by adding points in center-out order
    for p in order0:
        if p not in seed0 and _feasible_vec(p, seed0):
            seed0.append(p)
    # Local search to reduce
    seed0 = _local_search(seed0, max_iter=300)
    if len(seed0) < best_size:
        best_size = len(seed0)
        best_set = list(seed0)
    if time.time() > DEADLINE - 30:
        return best_set

    # Restart 1: row-major greedy
    order1 = _row_major_order()
    seed1 = _greedy(order1)
    for p in order1:
        if p not in seed1 and _feasible_vec(p, seed1):
            seed1.append(p)
    seed1 = _local_search(seed1, max_iter=300)
    if len(seed1) < best_size:
        best_size = len(seed1)
        best_set = list(seed1)
    if time.time() > DEADLINE - 30:
        return best_set

    # More random restarts
    restart = 2
    while time.time() < DEADLINE - 30:
        order_r = _random_order(rng)
        seed_r = _greedy(order_r)
        # Make maximal
        for p in order_r:
            if p not in seed_r and _feasible_vec(p, seed_r):
                seed_r.append(p)
        # Local search
        seed_r = _local_search(seed_r, max_iter=200)
        if len(seed_r) < best_size:
            best_size = len(seed_r)
            best_set = list(seed_r)
        restart += 1
        # Occasionally do a deeper local search on the current best
        if restart % 5 == 0 and best_set is not None:
            best_set = _local_search(best_set, max_iter=150)
            best_size = len(best_set)

    # Final safety: ensure best_set is valid and maximal
    if best_set is None:
        # Fallback: greedy center-out
        best_set = _greedy(_center_out_order())
        for p in _center_out_order():
            if p not in best_set and _feasible_vec(p, best_set):
                best_set.append(p)

    # Verify maximality one last time (with time guard)
    if time.time() < DEADLINE - 10:
        # If not maximal, add points to make it maximal
        order_final = _center_out_order()
        for p in order_final:
            if p not in best_set and _feasible_vec(p, best_set):
                best_set.append(p)

    return best_set
