# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 620.0  # safety margin before the 630s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    sq = (grid ** 2).sum(axis=1).astype(np.int64)  # squared norms

    def can_add(arr, p, psq):
        m = len(arr)
        if m < 4:
            return True
        # rows: (q - p, |q|^2 - |p|^2) for each existing point q
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = (arr ** 2).sum(axis=1) - psq
        if m == 4:
            return abs(np.linalg.det(w)) >= 0.5
        combos = list(itertools.combinations(range(m), 4))
        chunk = 4096
        for i in range(0, len(combos), chunk):
            idx = combos[i:i + chunk]
            sub = w[idx]
            if np.any(np.abs(np.linalg.det(sub)) < 0.5):
                return False
        return True

    def build(order):
        arr = np.empty((0, 3), dtype=np.int64)
        for p in order:
            if can_add(arr, p, (p ** 2).sum()):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
            if time.time() > deadline:
                break
        return arr

    def local_search(arr, max_iters=3000):
        arr = arr.copy()
        selected = set(map(tuple, arr.tolist()))
        unselected = [tuple(q) for q in grid if tuple(q) not in selected]
        improved = True
        it = 0
        while improved and it < max_iters and time.time() < deadline:
            improved = False
            it += 1
            for i in range(len(arr)):
                for cand in unselected:
                    cand = np.array(cand, dtype=np.int64)
                    rest = np.concatenate([arr[:i], arr[i + 1:]], axis=0)
                    if can_add(rest, cand, (cand ** 2).sum()):
                        arr = np.concatenate([rest, cand[None, :]], axis=0)
                        removed = tuple(arr[i - len(rest)])  # not needed, recompute
                        # track sets
                        # removed point is arr[i] from old; but we overwrote arr. Recompute:
                        # Actually removed = old arr[i]. We have rest = old without i.
                        # The removed point = the one at index i in old arr.
                        # We don't have it anymore easily; recompute from selected before update.
                        # Simpler: recompute selected/unselected from arr each iter (small).
                        selected = set(map(tuple, arr.tolist()))
                        unselected = [tuple(q) for q in grid if tuple(q) not in selected]
                        improved = True
                        break
                if improved:
                    break
        return arr

    rng = np.random.default_rng(0)
    best = None

    orderings = []
    for _ in range(4):
        orderings.append(grid[rng.permutation(N)])
    corners = np.array([[0,0,0],[7,7,7],[0,7,0],[7,0,7],[0,0,7],[7,7,0],[0,7,7],[7,0,0]], dtype=np.int64)
    corner_set = set(map(tuple, corners.tolist()))
    rest = np.array([q for q in grid if tuple(q) not in corner_set], dtype=np.int64)
    orderings.append(np.concatenate([corners, rest], axis=0))
    center = np.array([3, 3, 3])
    d = np.linalg.norm(grid - center, axis=1)
    orderings.append(grid[np.argsort(d)])
    dm = np.abs(grid - 3).sum(axis=1)
    orderings.append(grid[np.argsort(dm)])
    spread = grid.max(axis=1) - grid.min(axis=1)
    orderings.append(grid[np.argsort(-spread)])

    for order in orderings:
        if time.time() > deadline:
            break
        arr = build(order)
        arr = local_search(arr)
        if best is None or len(arr) > len(best):
            best = arr

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
