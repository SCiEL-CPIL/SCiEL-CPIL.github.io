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
    deadline = time.time() + 610.0  # safety margin before 630s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    grid_sq = (grid ** 2).sum(axis=1)  # precompute squared norms

    def can_add(arr: np.ndarray, p: np.ndarray, arr_sq: np.ndarray = None) -> bool:
        """Check if adding point p to arr keeps the set sphere-free."""
        m = len(arr)
        if m < 4:
            return True
        # Build lifted matrix: rows = (q - p, |q|^2 - |p|^2) for each q in arr
        diff = arr - p  # (m, 3)
        if arr_sq is None:
            arr_sq = (arr ** 2).sum(axis=1)
        p_sq = (p ** 2).sum()
        sq = arr_sq - p_sq  # (m,)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = diff
        w[:, 3] = sq
        # Check all 4-combinations
        if m < 25:
            combos = list(itertools.combinations(range(m), 4))
            if len(combos) == 0:
                return True
            sub = w[list(combos)]  # (C, 4, 4)
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)
        else:
            combos = list(itertools.combinations(range(m), 4))
            for i in range(0, len(combos), 1000):
                chunk = combos[i:i+1000]
                sub = w[list(chunk)]
                dets = np.linalg.det(sub)
                if np.any(np.abs(dets) < 0.5):
                    return False
            return True

    def greedy_pass(order: np.ndarray) -> np.ndarray:
        """Run one greedy pass with the given point ordering."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        for p in order:
            if can_add(arr, p, arr_sq):
                if len(arr) == 0:
                    arr = p[None, :].copy()
                    arr_sq = np.array([(p ** 2).sum()], dtype=np.float64)
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, (p ** 2).sum())
        return arr

    def polish(arr: np.ndarray) -> np.ndarray:
        """Try to add remaining points in sorted order."""
        selected = set(map(tuple, arr.tolist()))
        arr_sq = (arr ** 2).sum(axis=1) if len(arr) > 0 else np.empty(0)
        for p in grid:
            if tuple(p.tolist()) in selected:
                continue
            if can_add(arr, p, arr_sq):
                arr = np.vstack([arr, p])
                arr_sq = np.append(arr_sq, (p ** 2).sum())
                selected.add(tuple(p.tolist()))
        return arr

    def local_search(arr: np.ndarray, max_iters: int = 50) -> np.ndarray:
        """Try to swap selected points with unselected ones to improve."""
        selected = set(map(tuple, arr.tolist()))
        unselected = [p for p in grid if tuple(p.tolist()) not in selected]
        if not unselected:
            return arr

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 60:
            improved = False
            iters += 1
            for i in range(len(arr)):
                if time.time() > deadline - 60:
                    break
                # Try removing arr[i] and adding each unselected point
                arr_without_i = np.delete(arr, i, axis=0)
                arr_sq_without_i = np.delete((arr ** 2).sum(axis=1), i)
                for p in unselected:
                    if time.time() > deadline - 60:
                        break
                    if can_add(arr_without_i, p, arr_sq_without_i):
                        # Check if this swap improves (it does since we're adding a new point)
                        new_arr = np.vstack([arr_without_i, p])
                        new_sq = np.append(arr_sq_without_i, (p ** 2).sum())
                        # Verify the full set is still valid
                        # Actually we need to check that adding p to arr_without_i doesn't create
                        # any 5-point sphere with other points. can_add already checks this.
                        # But we also need to ensure the original set minus arr[i] plus p is valid.
                        # can_add(arr_without_i, p) checks if p can be added to arr_without_i,
                        # which means no 5 points in arr_without_i ∪ {p} are co-spherical.
                        # This is exactly what we need.
                        arr = new_arr
                        arr_sq = new_sq
                        selected.discard(tuple(arr[i].tolist()))
                        selected.add(tuple(p.tolist()))
                        unselected.remove(p)
                        improved = True
                        break
        return arr

    np.random.seed(0)
    best = None
    best_len = 0

    # Phase 1: Random greedy restarts
    num_restarts = 40
    for i in range(num_restarts):
        if time.time() > deadline - 90.0:
            break
        if i % 3 == 0:
            order = grid[np.random.permutation(N)]
        elif i % 3 == 1:
            # Distance from center ordering
            center = np.array([3.5, 3.5, 3.5])
            dists = np.linalg.norm(grid - center, axis=1)
            idx = np.argsort(dists + np.random.rand(N) * 0.1)
            order = grid[idx]
        else:
            # Random with slight bias toward corners
            corners = [np.array([0, 0, 0]), np.array([7, 7, 7]),
                       np.array([0, 7, 7]), np.array([7, 0, 7]),
                       np.array([0, 0, 7]), np.array([7, 7, 0]),
                       np.array([7, 0, 0]), np.array([0, 7, 0])]
            dists_to_corners = np.min([np.linalg.norm(grid - c, axis=1) for c in corners], axis=0)
            weights = 1.0 / (1.0 + dists_to_corners)
            weights /= weights.sum()
            idx = np.random.choice(N, size=N, replace=False, p=weights)
            order = grid[idx]
        arr = greedy_pass(order)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)

    # Phase 2: Deterministic polish on best
    if best is not None:
        best = polish(best)
        best_len = len(best)

    # Phase 3: Local search on best
    if best is not None and time.time() < deadline - 90:
        best = local_search(best, max_iters=100)
        best_len = len(best)

    # Phase 4: Try more restarts if time permits, with polish and local search
    for i in range(15):
        if time.time() > deadline - 60:
            break
        order = grid[np.random.permutation(N)]
        arr = greedy_pass(order)
        arr = polish(arr)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
        if time.time() < deadline - 60 and len(arr) >= best_len - 2:
            arr = local_search(arr, max_iters=50)
            if len(arr) > best_len:
                best = arr
                best_len = len(arr)

    # Final polish
    if best is not None and time.time() < deadline - 30:
        best = polish(best)
        best_len = len(best)

    if best is None:
        # Fallback: return a small valid set
        return grid[:10].astype(np.int64)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
