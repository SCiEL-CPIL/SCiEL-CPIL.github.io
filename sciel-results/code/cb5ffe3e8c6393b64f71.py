"""Kissing number in dimension 11 - geometric lattice approach via Construction A."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Strategy: Use the known optimal kissing configuration for d=11.
    # The kissing number in dimension 11 is 558 (or possibly 559).
    # We'll use a combination of vector types that are known to form optimal or near-optimal configurations.
    # 
    # For d=11, the optimal configuration uses:
    # - Type A: weight 4 vectors with entries in {-1, 0, 1} (norm^2 = 4)
    # - Type B: weight 1 vectors with entries in {-2, 0, 2} (norm^2 = 4)
    # 
    # The condition for non-overlap after normalization to norm 2:
    # <u,v> <= 2 for all pairs (since ||u||=||v||=2, so ||u||||v||=4, threshold=4/2=2)

    # Generate Type A: weight 4, entries in {-1,0,1}
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    # Generate Type B: weight 1, entries in {-2,0,2}
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute dot product matrix
    dots = all_candidates @ all_candidates.T
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists for efficiency
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def greedy_max_independent_set():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def local_search_improve(in_set_init, max_time=5.0):
        """Local search to improve independent set size."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        ls_deadline = time.time() + max_time

        # Count conflicts for each vertex
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0

        while time.time() < ls_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break

            counts = count_conflicts[not_in_set]
            zero_conflict = not_in_set[counts == 0]

            if len(zero_conflict) > 0:
                v = zero_conflict[rng.integers(len(zero_conflict))]
                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                # Try swap operations
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    break

                if rng.random() < 0.3:
                    u = in_set_indices[rng.integers(len(in_set_indices))]
                    potential_adds = np.where(conflicts[u] & ~in_set)[0]
                    if len(potential_adds) > 0:
                        v = potential_adds[rng.integers(len(potential_adds))]
                        v_conflicts_in_set = np.where(conflicts[v] & in_set)[0]
                        if len(v_conflicts_in_set) == 1 and v_conflicts_in_set[0] == u:
                            in_set[u] = False
                            in_set[v] = True
                            count_conflicts[neighbors[u]] -= 1
                            count_conflicts[neighbors[v]] += 1
                            stuck_counter += 1
                        else:
                            stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            if stuck_counter > 100:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 0:
                    u = in_set_indices[rng.integers(len(in_set_indices))]
                    in_set[u] = False
                    current_size -= 1
                    count_conflicts[neighbors[u]] -= 1
                    stuck_counter = 0

        return local_best_in_set, local_best_size

    # Multiple restarts with different strategies
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    # Strategy 1: Pure greedy with min-degree
    in_set = greedy_max_independent_set()
    current_size = in_set.sum()
    if current_size > best_size:
        best_size = current_size
        best_in_set = in_set.copy()

    # Strategy 2: Greedy + local search
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 2:
        ls_time = min(remaining_time * 0.4, 8.0)
        if ls_time >= 1:
            ls_result, ls_size = local_search_improve(best_in_set, max_time=ls_time)
            if ls_size > best_size:
                best_size = ls_size
                best_in_set = ls_result.copy()

    # Strategy 3: Multiple random restarts with local search
    n_restarts = 0
    max_restarts = 50
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_max_independent_set()
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        remaining_time = deadline - 8 - time.time()
        if remaining_time < 2:
            break

        ls_time = min(remaining_time * 0.3, 5.0)
        if ls_time < 1:
            ls_time = 1.0

        ls_result, ls_size = local_search_improve(in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]

    return result_vectors

# EVOLVE-BLOCK-END
