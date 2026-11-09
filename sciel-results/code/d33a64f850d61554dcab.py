# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 600.0  # safety margin before 630s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)  # |p|^2 for each grid point

    def can_add(arr: np.ndarray, p: np.ndarray, arr_sq: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        # Build lifted matrix: rows = (q_i - p, |q_i|^2 - |p|^2)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - (p ** 2).sum()
        
        # Vectorized: compute all C(m,4) determinants at once
        # For each 4-combination, form 4x4 matrix and take det
        # Use np.linalg.det on a (C, 4, 4) array
        idx = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int64)
        if len(idx) == 0:
            return True
        # Shape: (num_combos, 4, 4)
        mats = w[idx[:, :, None], idx]  # advanced indexing: idx has shape (C,4), idx[:,:,None] -> (C,4,1)
        # Actually we need: for each combo (i0,i1,i2,i3), matrix rows are w[i0], w[i1], w[i2], w[i3]
        # w[idx] gives shape (C, 4, 4) where idx has shape (C, 4)
        mats = w[idx]  # shape (C, 4, 4)
        dets = np.linalg.det(mats)
        if np.any(np.abs(dets) < 0.5):
            return False
        return True

    def greedy_fill(order: np.ndarray) -> np.ndarray:
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        rejected = []
        for idx in order:
            if time.time() > deadline - 5:
                break
            p = grid[idx]
            if can_add(arr, p, arr_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([grid_sq[idx]], dtype=np.float64)
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
            else:
                rejected.append(idx)
        
        # Local improvement: try to re-insert rejected points
        # Attempt multiple passes
        for pass_num in range(3):
            if time.time() > deadline - 10:
                break
            still_rejected = []
            for idx in rejected:
                if time.time() > deadline - 10:
                    still_rejected.append(idx)
                    continue
                p = grid[idx]
                if can_add(arr, p, arr_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
                else:
                    still_rejected.append(idx)
            if len(still_rejected) == len(rejected):
                break  # No progress
            rejected = still_rejected
        
        return arr

    best = np.empty((0, 3), dtype=np.int64)
    best_len = 0

    # Multiple restarts with different random seeds and heuristics
    rng = np.random.default_rng(42)
    num_restarts = 500  # Vectorized can_add is much faster, so more restarts feasible

    for restart in range(num_restarts):
        if time.time() > deadline - 30:  # 30s safety margin
            break

        # Vary the ordering strategy
        if restart % 3 == 0:
            # Pure random
            order = rng.permutation(len(grid))
        elif restart % 3 == 1:
            # Random with slight bias toward corners/edges (heuristic)
            # Points near corners have fewer neighbors, may be easier to add
            dist_from_center = np.sum((grid - 3.5) ** 2, axis=1)
            weights = dist_from_center + 1e-9  # Ensure strictly positive
            weights /= weights.sum()
            order = rng.choice(len(grid), size=len(grid), replace=False, p=weights)
        else:
            # Random with bias toward center (opposite heuristic)
            # Use inverse distance to bias towards center, ensuring positivity
            dist_from_center = np.sum((grid - 3.5) ** 2, axis=1)
            max_dist = dist_from_center.max()
            weights = (max_dist + 1.0 - dist_from_center) + 1e-9  # Ensure strictly positive
            weights /= weights.sum()
            order = rng.choice(len(grid), size=len(grid), replace=False, p=weights)

        result = greedy_fill(order)
        if len(result) > best_len:
            best = result.copy()
            best_len = len(result)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
