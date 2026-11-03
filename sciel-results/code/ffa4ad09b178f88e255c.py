"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    deadline = time.time() + 100

    # Generate norm^2=4 integer vectors in R^11
    candidates = []
    # Type A: single entry +/-2 (norm^2=4)
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    # Type B: four entries +/-1 (norm^2=4)
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)

    n = len(candidates)
    cand_arr = np.array(candidates)

    # Precompute conflict: two vectors conflict if their dot product > 2
    # (since ||vi-vj||^2 = 8 - 2*dot, need >= 4, so dot <= 2)
    dots = cand_arr @ cand_arr.T
    conflict = (dots > 2)
    np.fill_diagonal(conflict, False)

    best_size = 0
    best_set = None

    rng = np.random.RandomState(42)
    num_restarts = 300

    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        order = rng.permutation(n)
        selected = np.zeros(n, dtype=bool)
        count = 0

        for idx in order:
            if time.time() > deadline - 5:
                break
            if not np.any(conflict[idx] & selected):
                selected[idx] = True
                count += 1

        if count > best_size:
            best_size = count
            best_set = cand_arr[selected].copy()

        if best_size >= 550:
            break

    if best_set is not None:
        return best_set.astype(np.int64)

    # Fallback: seed construction
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
