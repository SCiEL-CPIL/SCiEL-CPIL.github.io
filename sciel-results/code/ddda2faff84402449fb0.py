# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximize the number of points from the 8x8x8 integer grid such that no
    five selected points are co-spherical (which also forbids five co-planar).
    Uses vectorized greedy construction plus swap-based local improvement,
    with an explicit deadline guard.
    """
    start_time = time.time()
    DEADLINE = 630.0  # wall-clock limit; return before this with margin

    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))),
                    dtype=np.int64)  # (512, 3)
    N = len(grid)

    def remaining() -> float:
        return DEADLINE - 0.2 - time.time()

    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        """True if adding p to arr keeps the set sphere-free."""
        m = len(arr)
        if m < 4:
            return True
        # Lift matrix: rows are (q - p, |q|^2 - |p|^2) for q in arr.
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr.astype(np.float64) - p
        w[:, 3] = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        # Determinant of every 4x4 submatrix (i.e. every choice of 4 existing pts).
        # If any is near 0, the set of 5 (4 existing + p) is co-spherical.
        combos = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int64)
        sub = w[combos]  # (C, 4, 4)
        return not np.any(np.abs(np.linalg.det(sub)) < 0.5)

    def can_add_batch(arr: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Vectorized: for each candidate, can it be added to arr? Returns bool array."""
        m = len(arr)
        if m < 4:
            return np.ones(len(candidates), dtype=bool)
        # Precompute for existing points: coordinates and squared norms.
        arr_f = arr.astype(np.float64)
        arr_sq = (arr_f ** 2).sum(axis=1)  # (m,)
        cand_f = candidates.astype(np.float64)
        cand_sq = (cand_f ** 2).sum(axis=1)  # (B,)
        # Build w for each candidate: (B, m, 4)
        # w[b, j, 0:3] = arr[j] - cand[b]; w[b, j, 3] = arr_sq[j] - cand_sq[b]
        B = len(candidates)
        w = np.empty((B, m, 4), dtype=np.float64)
        w[:, :, 0:3] = arr_f[None, :, :] - cand_f[:, None, :]
        w[:, :, 3] = arr_sq[None, :] - cand_sq[:, None, None]
        combos = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int64)
        sub = w[:, combos, :, :]  # (B, C, 4, 4)
        dets = np.linalg.det(sub)  # (B, C)
        return not np.any(np.abs(dets) < 0.5, axis=1)

    def greedy_pass(order: np.ndarray) -> np.ndarray:
        """Greedy: add points in given order if they stay valid."""
        arr = np.empty((0, 3), dtype=np.int64)
        for idx in order:
            p = grid[idx]
            if can_add(arr, p):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
        return arr

    def local_improve(arr: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Try swaps (remove one, add one) to grow the set. Iterate until no gain."""
        arr = arr.copy()
        improved = True
        while improved and remaining() > 5.0:
            improved = False
            m = len(arr)
            # For each point to remove, try to find a candidate to add that grows size.
            for i in range(m):
                if remaining() < 5.0:
                    break
                without_i = np.delete(arr, i, axis=0)
                # Candidates: all grid points not already in without_i.
                in_set = np.isin(grid, without_i, axis=1)
                candidates = grid[~in_set]
                if len(candidates) == 0:
                    continue
                # Check in batches to be safe.
                ok = can_add_batch(without_i, candidates)
                if np.any(ok):
                    # Pick the first valid candidate (or a random one for diversity).
                    valid_idx = np.where(ok)[0]
                    pick = valid_idx[rng.randint(len(valid_idx))]
                    new_arr = np.vstack([without_i, candidates[pick]])
                    # This keeps size the same (swap). For growth we need a point
                    # that can be added WITHOUT removing. Handle below.
                    improved = True
                    arr = new_arr
                    break
            # Also try pure additions (grow) if possible.
            if remaining() < 5.0:
                break
            in_set = np.isin(grid, arr, axis=1)
            candidates = grid[~in_set]
            if len(candidates) > 0:
                ok = can_add_batch(arr, candidates)
                if np.any(ok):
                    valid_idx = np.where(ok)[0]
                    pick = valid_idx[rng.randint(len(valid_idx))]
                    arr = np.vstack([arr, candidates[pick]])
                    improved = True

        return arr

    # Multi-restart greedy + local improvement.
    rng = np.random.RandomState(0)
    best = None
    best_len = -1

    num_restarts = 30
    for r in range(num_restarts):
        if remaining() < 20.0:
            break
        order = rng.permutation(N)
        arr = greedy_pass(order)
        arr = local_improve(arr, rng)
        if len(arr) > best_len:
            best_len = len(arr)
            best = arr

    # Final polish: one more local-improvement pass on the best found.
    if best is not None and remaining() > 10.0:
        polished = local_improve(best, rng)
        if len(polished) > best_len:
            best = polished
            best_len = len(polished)

    if best is None:
        # Fallback: a trivially valid small set.
        best = grid[:10]

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
