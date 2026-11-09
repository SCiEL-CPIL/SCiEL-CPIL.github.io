# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 580  # safety margin before 630s limit

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)  # |p|^2 for each grid point

    def can_add(arr, p, p_sq):
        """Check if point p (with |p|^2 = p_sq) can be added to arr without
        creating 5 co-spherical points. arr is (m,3) int array."""
        m = len(arr)
        if m < 4:
            return True
        # Lifted vectors: (q - p, |q|^2 - |p|^2)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = (arr ** 2).sum(axis=1) - p_sq

        # Check all combinations of 4 rows
        # For small m, use itertools; for larger m, vectorize
        if m <= 20:
            combos = list(itertools.combinations(range(m), 4))
            sub = w[combos]
            return not np.any(np.abs(np.linalg.det(sub)) < 0.5)
        else:
            # Vectorized: compute all C(m,4) determinants
            # This is expensive but we only call it when m is large
            combos = np.array(list(itertools.combinations(range(m), 4)))
            sub = w[combos]  # (C, 4, 4)
            return not np.any(np.abs(np.linalg.det(sub)) < 0.5)

    def greedy_pass(order, start_set=None):
        """Run one greedy pass over the given order of grid points."""
        if start_set is not None:
            arr = start_set.copy()
            arr_sq = (arr ** 2).sum(axis=1)
        else:
            arr = np.empty((0, 3), dtype=np.int64)
            arr_sq = np.empty(0, dtype=np.int64)
        
        for idx in order:
            if time.time() > deadline:
                break
            p = grid[idx]
            p_sq = grid_sq[idx]
            if can_add(arr, p, p_sq):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
                arr_sq = np.append(arr_sq, p_sq)
        return arr

    def local_improve(arr):
        """Try to insert previously-skipped points."""
        if len(arr) == 0:
            return arr
        arr_set = set(map(tuple, arr.tolist()))
        skipped = [grid[i] for i in range(len(grid)) if tuple(grid[i]) not in arr_set]
        improved = False
        # Try inserting each skipped point
        for p in skipped:
            if time.time() > deadline:
                break
            p_sq = (p ** 2).sum()
            if can_add(arr, p, p_sq):
                arr = np.vstack([arr, p])
                improved = True
        return arr

    np.random.seed(0)
    best = None
    best_len = 0

    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        order = np.random.permutation(len(grid))
        arr = greedy_pass(order)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
            # Try local improvement on the best
            if time.time() < deadline - 30:
                improved = local_improve(best)
                if len(improved) > best_len:
                    best = improved
                    best_len = len(improved)

    # Final local improvement pass on the best
    if best is not None and time.time() < deadline - 10:
        improved = local_improve(best)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)

    if best is None:
        # Fallback: return empty or a few points
        best = grid[:1]

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
