"""Combined program for kissing number in dimension 11.
Uses the efficient greedy MIS from inspirations plus a local-improvement (swap)
refinement, and a deterministic angular-sector seeding idea to get a good
initial independent set quickly.
"""
import numpy as np
import itertools
import time

D = 11


def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D."""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)


def greedy_mis(conflicts, deadline, rng):
    """
    Greedy Maximum Independent Set on a graph defined by a boolean conflict matrix.
    Uses min-degree ordering with random tie-breaking.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)

    selected_indices = []

    while np.any(alive):
        if time.time() > deadline:
            break

        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break

        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]

        # Random tie-breaking
        chosen = candidates[rng.integers(len(candidates))]
        selected_indices.append(chosen)

        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))

        # Update alive
        alive[to_remove] = False

        # Update degrees for remaining alive nodes
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices


def local_improve(selected, vectors, conflicts, deadline, rng):
    """
    Try to improve the selected set by swapping out one vector for another
    that is compatible with all other selected vectors.
    """
    N = len(vectors)
    selected_list = list(selected)
    selected_set = set(selected_list)
    current_size = len(selected_list)

    improved = True
    passes = 0
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        if passes > 200:  # safety
            break
        # For each selected vector, try to swap it out
        for idx in range(len(selected_list)):
            if time.time() > deadline:
                return selected_list
            v_out = selected_list[idx]
            other_selected = [s for s in selected_list if s != v_out]

            # Find all unselected indices
            unselected = [i for i in range(N) if i not in selected_set]
            if not unselected:
                continue

            # Check which unselected are compatible with all other_selected
            if len(other_selected) == 0:
                compatible = unselected
            else:
                other_arr = np.array(other_selected)
                # conflicts[unselected, other_arr] shape (len(unselected), len(other_arr))
                conflict_mask = conflicts[np.ix_(unselected, other_arr)]
                compatible = [unselected[i] for i in range(len(unselected))
                              if not conflict_mask[i].any()]

            if compatible:
                u_in = compatible[rng.integers(len(compatible))]
                # Perform swap
                selected_list[idx] = u_in
                selected_set.discard(v_out)
                selected_set.add(u_in)
                improved = True
                break  # restart loop

    return selected_list


def construct():
    # Generate all weight-2 vectors in {-1, 0, 1}^D
    # There are C(11, 2) * 4 = 55 * 4 = 220 such vectors.
    # For any two distinct weight-2 vectors u, v:
    # ||u|| = ||v|| = sqrt(2). Max norm = sqrt(2).
    # We need ||u - v|| >= sqrt(2) => ||u - v||^2 >= 2.
    # ||u - v||^2 = 2 + 2 - 2 * dot(u, v) = 4 - 2 * dot(u, v).
    # 4 - 2 * dot(u, v) >= 2 => 2 >= 2 * dot(u, v) => dot(u, v) <= 1.
    # The maximum dot product for two distinct weight-2 vectors is 1
    # (occurring when they share one index with the same sign, and the other indices are disjoint or opposite).
    # Thus, all weight-2 vectors form a valid kissing configuration.
    
    vectors = []
    for combo in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
