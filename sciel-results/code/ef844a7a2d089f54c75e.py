# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 620

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    N = len(grid)  # 512

    def check_add(arr_pts, arr_sq, p):
        """Check if adding point p to arr_pts keeps all 5-tuples non-degenerate.
        Returns True if valid, False otherwise."""
        m = len(arr_pts)
        if m < 4:
            return True
        p = np.asarray(p, dtype=np.float64)
        p_sq = float(np.dot(p, p))
        # Build lifted vectors
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr_pts - p
        w[:, 3] = arr_sq - p_sq
        # Check all 4-combinations
        for combo in itertools.combinations(range(m), 4):
            d = abs(np.linalg.det(w[list(combo)]))
            if d < 0.5:
                return False
        return True

    def min_det_value(arr_pts, arr_sq, p):
        """Return minimum |det| over all 4-tuples, or inf if m<4."""
        m = len(arr_pts)
        if m < 4:
            return np.inf
        p = np.asarray(p, dtype=np.float64)
        p_sq = float(np.dot(p, p))
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr_pts - p
        w[:, 3] = arr_sq - p_sq
        min_d = np.inf
        for combo in itertools.combinations(range(m), 4):
            d = abs(np.linalg.det(w[list(combo)]))
            if d < min_d:
                min_d = d
        return min_d

    def greedy_random_order(seed, time_limit):
        """Simple greedy with random order."""
        np.random.seed(seed)
        order = np.random.permutation(N)
        arr_pts = np.empty((0, 3), dtype=np.float64)
        arr_sq = np.empty(0, dtype=np.float64)
        start = time.time()
        for idx in order:
            if time.time() - start > time_limit:
                break
            p = grid[idx]
            if check_add(arr_pts, arr_sq, p):
                arr_pts = np.vstack([arr_pts, p]) if len(arr_pts) > 0 else p[None, :]
                arr_sq = np.append(arr_sq, grid_sq[idx])
        return arr_pts

    def greedy_best_first(seed, time_limit):
        """Greedy that picks the point with largest min_det at each step."""
        np.random.seed(seed)
        remaining = list(range(N))
        arr_pts = np.empty((0, 3), dtype=np.float64)
        arr_sq = np.empty(0, dtype=np.float64)
        start = time.time()
        
        while remaining:
            if time.time() - start > time_limit:
                break
            if len(arr_pts) < 4:
                # Just pick randomly
                idx = np.random.choice(remaining)
            else:
                # Sample candidates
                sample_size = min(len(remaining), 80)
                candidates = np.random.choice(remaining, size=sample_size, replace=False)
                best_idx = None
                best_md = -1
                for idx in candidates:
                    md = min_det_value(arr_pts, arr_sq, grid[idx])
                    if md >= 0.5 and md > best_md:
                        best_md = md
                        best_idx = idx
                if best_idx is None:
                    # Try all remaining if sample failed
                    best_idx = None
                    for idx in remaining:
                        md = min_det_value(arr_pts, arr_sq, grid[idx])
                        if md >= 0.5:
                            if best_idx is None:
                                best_idx = idx
                            break
                if best_idx is None:
                    break
            
            arr_pts = np.vstack([arr_pts, grid[best_idx]]) if len(arr_pts) > 0 else grid[best_idx][None, :]
            arr_sq = np.append(arr_sq, grid_sq[best_idx])
            remaining.remove(best_idx)
        
        return arr_pts

    # Run multiple strategies
    best_pts = None
    best_len = 0

    # Strategy 1: Random order greedy with many seeds
    for i in range(20):
        if time.time() > deadline - 100:
            break
        t_left = min(15, deadline - time.time() - 100)
        pts = greedy_random_order(seed=i * 137 + 7, time_limit=t_left)
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts
            print(f"Random greedy {i}: {len(pts)}")

    # Strategy 2: Best-first greedy
    for i in range(8):
        if time.time() > deadline - 80:
            break
        t_left = min(40, deadline - time.time() - 80)
        pts = greedy_best_first(seed=i * 251 + 13, time_limit=t_left)
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts
            print(f"Best-first greedy {i}: {len(pts)}")

    # Strategy 3: Local search / hill climbing on best solution
    if best_pts is not None and time.time() < deadline - 30:
        np.random.seed(12345)
        start = time.time()
        improved = True
        while improved and time.time() - start < deadline - time.time() - 20:
            improved = False
            # Try to add random points
            in_set = set(map(tuple, best_pts.astype(int)))
            for _ in range(100):
                if time.time() - start > deadline - time.time() - 20:
                    break
                idx = np.random.randint(N)
                p = grid[idx]
                if tuple(p) not in in_set:
                    if check_add(best_pts, (best_pts**2).sum(axis=1), p):
                        best_pts = np.vstack([best_pts, p])
                        in_set.add(tuple(p))
                        best_len = len(best_pts)
                        improved = True
                        break
            # Try swaps: remove one, add two
            if len(best_pts) >= 2:
                for _ in range(50):
                    if time.time() - start > deadline - time.time() - 20:
                        break
                    rem_idx = np.random.randint(len(best_pts))
                    new_pts = np.delete(best_pts, rem_idx, axis=0)
                    new_sq = (new_pts**2).sum(axis=1)
                    in_set2 = set(map(tuple, new_pts.astype(int)))
                    added = 0
                    for _ in range(10):
                        idx = np.random.randint(N)
                        p = grid[idx]
                        if tuple(p) not in in_set2:
                            if check_add(new_pts, new_sq, p):
                                new_pts = np.vstack([new_pts, p])
                                new_sq = np.append(new_sq, (p**2).sum())
                                in_set2.add(tuple(p))
                                added += 1
                    if added > 1:
                        best_pts = new_pts
                        best_len = len(best_pts)
                        improved = True
                        break

    if best_pts is None:
        return np.empty((0, 3), dtype=np.int64)

    return best_pts.astype(np.int64)


# EVOLVE-BLOCK-END
