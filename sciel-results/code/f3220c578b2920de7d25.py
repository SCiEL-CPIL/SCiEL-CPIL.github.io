"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
import sys


def construct():
    deadline = time.time() + 140

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

    # Convert conflict to bitmasks for fast operations
    # Use Python integers as bitmasks
    conflict_masks = []
    for i in range(n):
        mask = 0
        row = conflict[i]
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        conflict_masks.append(mask)

    best_size = 0
    best_set = None

    rng = np.random.RandomState(42)
    num_restarts = 200

    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break

        # Greedy with random ordering
        order = rng.permutation(n)
        selected_mask = 0
        selected_indices = []

        for idx in order:
            if time.time() > deadline - 5:
                break
            # Check if idx conflicts with any selected
            if (selected_mask & (1 << idx)) == 0 and not (conflict_masks[idx] & selected_mask):
                selected_mask |= (1 << idx)
                selected_indices.append(idx)

        count = len(selected_indices)

        if count > best_size:
            best_size = count
            best_set = cand_arr[selected_indices].copy()

        if best_size >= 550:
            break

    # Local search: try to improve the best set found
    if best_set is not None and time.time() < deadline - 10:
        # Build a conflict matrix for the selected set
        sel_count = len(best_set)
        sel_conflict = np.zeros((sel_count, sel_count), dtype=bool)
        for i in range(sel_count):
            for j in range(sel_count):
                if i != j:
                    sel_conflict[i, j] = conflict[best_set[i] // 2, best_set[j] // 2] if best_set[i] < n else False

        # Actually, let's do a simpler local search: try swapping out one vector for another
        selected_set = set(best_set.tolist())
        selected_mask2 = 0
        for idx in selected_set:
            selected_mask2 |= (1 << idx)

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each selected vector and adding a non-selected one
            for rem_idx in list(selected_set):
                if time.time() > deadline - 5:
                    break
                # Remove rem_idx
                temp_mask = selected_mask2 & ~(1 << rem_idx)
                # Try adding each non-selected vector
                for add_idx in range(n):
                    if time.time() > deadline - 5:
                        break
                    if add_idx in selected_set:
                        continue
                    if (temp_mask & (1 << add_idx)) != 0:
                        continue
                    if conflict_masks[add_idx] & temp_mask:
                        continue
                    # This swap works
                    selected_set.discard(rem_idx)
                    selected_set.add(add_idx)
                    selected_mask2 = temp_mask | (1 << add_idx)
                    improved = True
                    break

        if len(selected_set) > best_size:
            best_size = len(selected_set)
            best_set = cand_arr[list(selected_set)].copy()

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
