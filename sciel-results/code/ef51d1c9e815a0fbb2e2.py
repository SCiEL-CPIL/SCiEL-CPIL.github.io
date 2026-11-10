# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """Select as many points as possible from {0..7}^3 with no 5 co-spherical."""
    n = 8
    deadline = time.time() + 600.0
    safety = 20.0

    grid = np.array(list(itertools.product(range(n), range(n), range(n))),
                    dtype=np.int64)
    N = len(grid)
    sq = (grid ** 2).sum(axis=1)

    def det_check(arr_idx: np.ndarray, p_idx: int) -> bool:
        """Check if p_idx can be added to arr_idx without creating 5 co-spherical."""
        m = len(arr_idx)
        if m < 4:
            return True
        a = grid[arr_idx]
        p = grid[p_idx]
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = a - p
        w[:, 3] = sq[arr_idx] - sq[p_idx]
        if m == 4:
            return abs(np.linalg.det(w)) >= 0.5
        combos = np.array(list(itertools.combinations(range(m), 4)))
        sub = w[combos]
        dets = np.linalg.det(sub)
        return bool(np.all(np.abs(dets) >= 0.5))

    # Phase 1: multiple random greedy restarts
    best_arr = np.array([], dtype=np.int64)
    rng = np.random.default_rng(0)

    for restart in range(30):
        if time.time() > deadline - safety:
            break
        order = rng.permutation(N)
        arr = []
        for p_idx in order:
            if det_check(np.array(arr, dtype=np.int64), p_idx):
                arr.append(p_idx)
        if len(arr) > len(best_arr):
            best_arr = np.array(arr, dtype=np.int64)

    # Phase 2: local search - try to add any remaining point
    S = list(best_arr)
    S_set = set(S)
    for c in range(N):
        if c in S_set:
            continue
        if det_check(np.array(S, dtype=np.int64), c):
            S.append(c)
            S_set.add(c)
    best_arr = np.array(S, dtype=np.int64)
    S_set = set(S)

    # Phase 3: swap-based local search with time limit
    improved = True
    while improved and time.time() < deadline - safety:
        improved = False
        S = list(best_arr)
        S_set = set(S)
        unselected = [i for i in range(N) if i not in S_set]
        selected = list(S)

        rng_ls = np.random.default_rng(42)
        # Randomize order of unselected and selected for diversity
        rng_ls.shuffle(unselected)
        rng_ls.shuffle(selected)

        for c in unselected:
            if time.time() > deadline - safety:
                break
            for s in selected:
                if time.time() > deadline - safety:
                    break
                # Try swap: remove s, add c
                new_S = [x for x in S if x != s]
                if det_check(np.array(new_S, dtype=np.int64), c):
                    best_arr = np.array(new_S + [c], dtype=np.int64)
                    improved = True
                    break
            if improved:
                break

        # Try additions again
        if not improved:
            S = list(best_arr)
            S_set = set(S)
            for c in range(N):
                if c in S_set:
                    continue
                if det_check(np.array(S, dtype=np.int64), c):
                    S.append(c)
                    S_set.add(c)
                    best_arr = np.array(S, dtype=np.int64)
                    improved = True

    return best_arr.astype(np.int64)


# EVOLVE-BLOCK-END
