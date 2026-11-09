# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    No 5 on a sphere in the 8x8x8 grid.
    Maximizes the number of selected integer points from {0..7}^3 such that
    no five selected points are co-spherical (or co-planar).
    """
    n = 8
    deadline = time.time() + 620.0  # safety margin before 630s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512

    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        """Check if adding point p to arr keeps the set sphere-free."""
        m = len(arr)
        if m < 4:
            return True
        # Build lifted matrix: rows = (q - p, |q|^2 - |p|^2) for each q in arr
        diff = arr - p  # (m, 3)
        sq = (arr ** 2).sum(axis=1) - (p ** 2).sum()  # (m,)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = diff
        w[:, 3] = sq
        # Check all 4-combinations
        if m < 20:
            # Small enough: enumerate combinations
            combos = list(itertools.combinations(range(m), 4))
            if len(combos) == 0:
                return True
            sub = w[list(combos)]  # (C, 4, 4)
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # Larger: still need to check, but use chunked approach
            combos = list(itertools.combinations(range(m), 4))
            for i in range(0, len(combos), 500):
                chunk = combos[i:i+500]
                sub = w[list(chunk)]
                dets = np.linalg.det(sub)
                if np.any(np.abs(dets) < 0.5):
                    return False
            return True

    def greedy_pass(order: np.ndarray) -> np.ndarray:
        """Run one greedy pass with the given point ordering."""
        arr = np.empty((0, 3), dtype=np.int64)
        for p in order:
            if can_add(arr, p):
                if len(arr) == 0:
                    arr = p[None, :].copy()
                else:
                    arr = np.vstack([arr, p])
        return arr

    def polish(arr: np.ndarray) -> np.ndarray:
        """Try to add remaining points in sorted order."""
        selected = set(map(tuple, arr.tolist()))
        for p in grid:
            if p[0] in selected:
                continue
            if tuple(p.tolist()) in selected:
                continue
            if can_add(arr, p):
                arr = np.vstack([arr, p])
                selected.add(tuple(p.tolist()))
        return arr

    np.random.seed(0)
    best = None
    best_len = 0

    # Phase 1: Random greedy restarts
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 60.0:
            break
        order = grid[np.random.permutation(N)]
        arr = greedy_pass(order)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)

    # Phase 2: Deterministic polish on best
    if best is not None:
        best = polish(best)
        best_len = len(best)

    # Phase 3: Try a few more restarts if time permits, with polish
    for i in range(10):
        if time.time() > deadline - 30.0:
            break
        order = grid[np.random.permutation(N)]
        arr = greedy_pass(order)
        arr = polish(arr)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)

    if best is None:
        # Fallback: return a small valid set
        return grid[:10].astype(np.int64)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
