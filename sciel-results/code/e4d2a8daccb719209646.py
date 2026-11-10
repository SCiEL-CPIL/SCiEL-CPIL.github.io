# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    The "no 5 on a sphere" problem. Select as many points as possible from the
    n x n x n integer grid {0,...,n-1}^3 such that no five of the selected points
    are co-spherical (lie on a common sphere, which also forbids five co-planar
    points). Five points p_i = (x_i, y_i, z_i) are co-spherical iff the 5x5
    determinant of rows [x_i, y_i, z_i, x_i^2 + y_i^2 + z_i^2, 1] vanishes.

    Here n = 8. The objective is to MAXIMIZE the number of selected points.

    Strategy:
      1. Randomized greedy: shuffle grid, add points if they don't create
         5 co-spherical points.
      2. Local improvement: try to swap out a point for a different one that
         increases the total count or maintains count with better "slack".
      3. Multiple restarts with time budget.

    Returns:
        points: np.ndarray of shape (k, 3) of integer grid coordinates in
            [0, n-1], no five of which are co-spherical.
    """
    n = 8
    deadline = time.time() + 570  # 30s safety margin

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512

    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        """Check if adding point p to arr keeps no 5 co-spherical."""
        m = len(arr)
        if m < 4:
            return True
        # Lift each existing point to a 4-vector relative to the candidate p.
        # The candidate together with four existing points is co-spherical iff
        # the corresponding four lifted vectors are linearly dependent.
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        # Check all 4-combinations for zero determinant
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    def greedy_pass(rng):
        """Run one greedy pass with given RNG state."""
        order = grid[rng.permutation(N)]
        arr = np.empty((0, 3), dtype=np.int64)
        for p in order:
            if can_add(arr, p):
                if len(arr) == 0:
                    arr = p[None, :]
                else:
                    arr = np.vstack([arr, p])
        return arr

    def local_improve(arr, rng, max_iters=200):
        """Try to improve arr by swapping points."""
        arr = arr.copy()
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 30:
            improved = False
            iters += 1
            m = len(arr)
            # Try each point in arr
            for i in range(m):
                if time.time() >= deadline - 30:
                    break
                # Try removing point i
                arr_no_i = np.delete(arr, i, axis=0)
                # Try adding each grid point not already in arr
                for j in range(N):
                    if time.time() >= deadline - 30:
                        break
                    p = grid[j]
                    # Check if p is already in arr (excluding i)
                    if i < m and np.array_equal(p, arr[i]):
                        continue
                    # Check if p is in arr_no_i
                    if any(np.array_equal(p, arr_no_i[k]) for k in range(len(arr_no_i))):
                        continue
                    if can_add(arr_no_i, p):
                        # This swap maintains the same size; accept with some probability
                        # or if it leads to further improvement
                        new_arr = np.vstack([arr_no_i, p])
                        # Accept the swap (it keeps size same but may enable future swaps)
                        arr = new_arr
                        improved = True
                        break
                if improved:
                    break
        return arr

    def greedy_with_local(arr, rng, max_iters=100):
        """After greedy, try to add more points from remaining grid."""
        arr = arr.copy()
        for _ in range(max_iters):
            if time.time() >= deadline - 30:
                break
            added = False
            for j in range(N):
                if time.time() >= deadline - 30:
                    break
                p = grid[j]
                if any(np.array_equal(p, arr[k]) for k in range(len(arr))):
                    continue
                if can_add(arr, p):
                    arr = np.vstack([arr, p])
                    added = True
                    break
            if not added:
                break
        return arr

    np.random.seed(42)
    rng = np.random.RandomState(42)
    best = None
    best_len = 0

    # Phase 1: Multiple greedy restarts
    restart_count = 0
    while time.time() < deadline - 60:
        arr = greedy_pass(rng)
        arr = greedy_with_local(arr, rng)
        restart_count += 1
        if best is None or len(arr) > best_len:
            best = arr
            best_len = len(arr)

    # Phase 2: Local improvement on best solution
    if best is not None and time.time() < deadline - 30:
        best = local_improve(best, rng, max_iters=500)

    # Phase 3: Try more restarts if time remains
    while time.time() < deadline - 30:
        arr = greedy_pass(rng)
        arr = greedy_with_local(arr, rng)
        if len(arr) > len(best):
            best = arr
            best_len = len(arr)
            # Try local improvement on new best
            if time.time() < deadline - 30:
                best = local_improve(best, rng, max_iters=200)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
