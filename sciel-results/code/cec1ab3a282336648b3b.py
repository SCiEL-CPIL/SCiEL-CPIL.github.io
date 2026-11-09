# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs 11-dimensional integer points using the complete set of
    ±1,±1 vectors on pairs of coordinates (220 vectors of norm²=2),
    then greedily extends with higher-norm vectors using vectorized checks.
    
    All 220 norm²=2 vectors are mutually compatible since min pairwise dist²=2.
    """
    d = 11
    start_time = time.time()
    deadline = 600.0

    # Generate all 220 vectors with exactly two ±1 coordinates
    points_list = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points_list.append(v)

    points_arr = np.array(points_list, dtype=np.int64)
    R2 = 2  # current max norm squared

    def _try_add_candidates(candidates, norm_sq):
        """Try to add each candidate vector, vectorized compatibility check."""
        nonlocal points_arr, R2
        added = 0
        for v in candidates:
            if time.time() > deadline - 30:
                break
            v = np.asarray(v, dtype=np.int64)
            dots = points_arr @ v
            p2 = np.sum(points_arr ** 2, axis=1)
            dist_sq = norm_sq + p2 - 2 * dots
            required = max(R2, norm_sq)
            if np.all(dist_sq >= required):
                points_arr = np.vstack([points_arr, v.reshape(1, -1)])
                R2 = max(R2, norm_sq)
                added += 1
        return added

    # Try adding norm²=3 vectors (3 nonzero ±1 coords): 165*8 = 1320 candidates
    if time.time() < deadline - 90:
        added3 = 0
        for combo in itertools.combinations(range(d), 3):
            if time.time() > deadline - 30:
                break
            for signs in itertools.product([1, -1], repeat=3):
                v = np.zeros(d, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                dots = points_arr @ v
                p2 = np.sum(points_arr ** 2, axis=1)
                dist_sq = 3 + p2 - 2 * dots
                required = max(R2, 3)
                if np.all(dist_sq >= required):
                    points_arr = np.vstack([points_arr, v.reshape(1, -1)])
                    R2 = 3
                    added3 += 1
        print(f"Added {added3} norm²=3 vectors. Total: {len(points_arr)}")

    # Try norm²=4 vectors (4 nonzero ±1 coords): 330*16 = 5280 candidates
    if time.time() < deadline - 120:
        added4 = 0
        for combo in itertools.combinations(range(d), 4):
            if time.time() > deadline - 30:
                break
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(d, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                dots = points_arr @ v
                p2 = np.sum(points_arr ** 2, axis=1)
                dist_sq = 4 + p2 - 2 * dots
                required = max(R2, 4)
                if np.all(dist_sq >= required):
                    points_arr = np.vstack([points_arr, v.reshape(1, -1)])
                    R2 = 4
                    added4 += 1
        print(f"Added {added4} norm²=4 vectors. Total: {len(points_arr)}")

    # Final validation
    k_final = len(points_arr)
    if k_final >= 2:
        result = np.around(points_arr).astype(np.int64)
        norms_sq = np.sum(result ** 2, axis=1)
        if np.all(norms_sq > 0):
            max_norm_sq = int(np.max(norms_sq))
            min_dist_sq = float('inf')
            for i in range(k_final):
                for j in range(i + 1, k_final):
                    diff = result[i] - result[j]
                    dist_sq = int(np.sum(diff ** 2))
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
            if min_dist_sq >= max_norm_sq:
                print(f"VALID: k={k_final}, max_norm²={max_norm_sq}, min_dist²={min_dist_sq}")
                return result
            else:
                print(f"INVALID: min_dist²={min_dist_sq} < max_norm²={max_norm_sq}")

    # Fallback: return the 220 norm²=2 vectors
    base = np.array(points_list, dtype=np.int64)
    return np.around(base).astype(np.int64)

# EVOLVE-BLOCK-END
