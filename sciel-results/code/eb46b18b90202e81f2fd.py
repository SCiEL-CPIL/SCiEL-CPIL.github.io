import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = 580

    def fallback():
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        points.append(v)
        return np.array(points, dtype=np.int64)

    try:
        all_vectors = []
        for support in combinations(range(d), 4):
            for signs in range(16):
                v = np.zeros(d, dtype=np.int64)
                for k_idx, pos in enumerate(support):
                    v[pos] = 1 if (signs >> k_idx) & 1 else -1
                all_vectors.append(v)

        mat = np.array(all_vectors)
        n = len(mat)

        dots = mat @ mat.T
        conflict = dots > 2

        available = np.ones(n, dtype=bool)
        selected = []

        while available.any():
            if time.time() - start_time > deadline:
                break
            conflict_counts = conflict[:, available].sum(axis=1)
            conflict_counts[~available] = n + 1
            idx = int(np.argmin(conflict_counts))
            selected.append(idx)
            avail_conflicts = conflict[idx] & available
            available[avail_conflicts] = False
            available[idx] = False

        if len(selected) > 220:
            return mat[selected].astype(np.int64)

    except Exception:
        pass

    return fallback()
