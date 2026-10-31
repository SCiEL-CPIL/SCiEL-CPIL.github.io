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
    t0 = time.time()
    deadline = t0 + 115  # Safety margin before 150s kill

    # Generate weight-4 vectors
    vectors = generate_weight_4_vectors()
    N = len(vectors)

    if N == 0:
        return np.empty((0, D), dtype=np.int64)

    # Compute dot products
    dots = vectors @ vectors.T

    # For weight-4 vectors, norm^2 = 4.
    # We need dist^2 >= max_norm^2. If all vectors have norm^2=4, max_norm^2=4.
    # dist^2 = 4 + 4 - 2*dot = 8 - 2*dot >= 4  =>  dot <= 2
    # So conflict if dot > 2
    conflicts = (dots > 2).astype(np.int8)

    # Self-conflicts don't matter since we pick distinct vectors
    np.fill_diagonal(conflicts, 0)

    best_set = []
    rng = np.random.default_rng(42)

    # Run multiple greedy MIS restarts
    n_restarts = 0
    while time.time() < deadline - 20:  # Leave 20s for local improvement
        current_set = greedy_mis(conflicts, deadline - 20, rng)
        n_restarts += 1

        if len(current_set) > len(best_set):
            best_set = current_set

        # Occasionally do local improvement on the best so far
        if n_restarts % 5 == 0 and len(best_set) > 0:
            improved = local_improve(best_set, vectors, conflicts,
                                     min(deadline - 5, time.time() + 3), rng)
            if len(improved) > len(best_set):
                best_set = improved

    # Final local improvement pass
    if len(best_set) > 0 and time.time() < deadline - 5:
        improved = local_improve(best_set, vectors, conflicts, deadline - 5, rng)
        if len(improved) > len(best_set):
            best_set = improved

    if not best_set:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    result_vectors = vectors[best_set]
    return result_vectors.astype(np.int64)
