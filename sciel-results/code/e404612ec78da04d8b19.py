# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Select as many points from the 8x8x8 grid {0,...,7}^3 such that no five
    are co-spherical.
    """
    n = 8
    deadline = time.time() + 600

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    sq_norms = (grid.astype(np.float64) ** 2).sum(axis=1)

    def can_add(arr: np.ndarray, p: np.ndarray, p_sq: float, arr_sq: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    def greedy_with_order(order: np.ndarray, order_sq: np.ndarray) -> np.ndarray:
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        for i in range(len(order)):
            p = order[i]
            p_sq = order_sq[i]
            if can_add(arr, p, p_sq, arr_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = p_sq[None, :]
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
        return arr

    np.random.seed(42)
    best = np.empty((0, 3), dtype=np.int64)

    # Phase 1: Random restarts
    for i in range(40):
        if time.time() > deadline - 30:
            break
        perm = np.random.permutation(N)
        order = grid[perm]
        order_sq = sq_norms[perm]
        arr = greedy_with_order(order, order_sq)
        if len(arr) > len(best):
            best = arr

    # Phase 2: Heuristic orderings
    center = np.array([3.5, 3.5, 3.5])
    heuristics = [
        np.linalg.norm(grid - center, axis=1),
        -np.linalg.norm(grid - center, axis=1),
        np.abs(grid - 3.5).sum(axis=1),
        -np.abs(grid - 3.5).sum(axis=1),
        grid.max(axis=1),
        -grid.min(axis=1),
        (grid % 2).sum(axis=1),
    ]
    for h in heuristics:
        if time.time() > deadline - 30:
            break
        idx = np.argsort(h, kind='stable')
        order = grid[idx]
        order_sq = sq_norms[idx]
        arr = greedy_with_order(order, order_sq)
        if len(arr) > len(best):
            best = arr

    # Phase 3: More random restarts
    rng = np.random.RandomState(123)
    for i in range(100):
        if time.time() > deadline - 30:
            break
        perm = rng.permutation(N)
        order = grid[perm]
        order_sq = sq_norms[perm]
        arr = greedy_with_order(order, order_sq)
        if len(arr) > len(best):
            best = arr

    # Phase 4: Local search
    if time.time() < deadline - 60:
        k = len(best)
        best_set = set(map(tuple, best.tolist()))
        remaining = [tuple(g) for g in grid if tuple(g) not in best_set]
        rng2 = np.random.RandomState(777)

        improved = True
        while improved and time.time() < deadline - 60:
            improved = False
            for idx in range(min(k, 30)):
                if time.time() > deadline - 60:
                    break
                trial = np.delete(best, idx, axis=0)
                trial_sq = (trial.astype(np.float64) ** 2).sum(axis=1) if len(trial) > 0 else np.empty(0)
                best_trial_set = set(map(tuple, trial.tolist()))
                for cand in remaining:
                    if len(trial) >= 500:
                        break
                    if cand in best_trial_set:
                        continue
                    p = np.array(cand, dtype=np.float64)
                    p_sq = (p ** 2).sum()
                    if can_add(trial, p, p_sq, trial_sq):
                        trial = np.vstack([trial, p])
                        trial_sq = np.append(trial_sq, p_sq)
                        best_trial_set.add(cand)
                        remaining.remove(cand)
                        improved = True
                        break
                if len(trial) > len(best):
                    best = trial
                    best_set = set(map(tuple, best.tolist()))
                    remaining = [tuple(g) for g in grid if tuple(g) not in best_set]
                    improved = True
                    break

    # Phase 5: More random restarts
    if time.time() < deadline - 30:
        rng3 = np.random.RandomState(999)
        for i in range(200):
            if time.time() > deadline - 30:
                break
            perm = rng3.permutation(N)
            order = grid[perm]
            order_sq = sq_norms[perm]
            arr = greedy_with_order(order, order_sq)
            if len(arr) > len(best):
                best = arr

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
