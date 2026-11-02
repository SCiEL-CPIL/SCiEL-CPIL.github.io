"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    # Precompute conflict rows as boolean arrays for fast operations
    conflict_rows = [conflicts[i] for i in range(N4)]

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
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

    def sa_local_search(in_set_init, max_time=10.0):
        """Simulated annealing local search with cached not_in_set."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 3.0
        cooling = 0.9995
        min_T = 0.05

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        # Cache not-in-set indices
        not_in_set = np.where(~in_set)[0]

        while time.time() < sa_deadline:
            if len(not_in_set) == 0:
                break

            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)

            if min_count == 0:
                # Add a 0-conflict node
                zero_cands = not_in_set[counts == 0]
                v = zero_cands[rng.integers(len(zero_cands))]

                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1

                # Remove v from not_in_set
                mask = not_in_set != v
                not_in_set = not_in_set[mask]

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue

            # SA step: pick node with minimum conflict count
            min_cands = not_in_set[counts == min_count]
            v = min_cands[rng.integers(len(min_cands))]

            k = min_count
            delta = 1 - k

            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                if delta < -20 * T:
                    accept = False
                else:
                    accept = rng.random() < np.exp(delta / T)

            if accept:
                # Remove neighbors of v that are in set
                to_remove = np.where(conflict_rows[v] & in_set)[0]

                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1

                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1

                # Update not_in_set: remove v, add back to_remove
                if len(to_remove) > 0:
                    mask = not_in_set != v
                    not_in_set = np.concatenate([not_in_set[mask], to_remove])
                else:
                    mask = not_in_set != v
                    not_in_set = not_in_set[mask]

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1

            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def exact_polish(in_set_init, max_time=5.0):
        """Exact local search: try all single additions and 1-swap and 2-swap moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        improved = True
        while improved and time.time() < polish_deadline:
            improved = False

            # Try adding nodes with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                zero_cands = not_in_set[counts == 0]
                for v in zero_cands:
                    in_set[v] = True
                    current_size += 1
                    count_conflicts[neighbors[v]] += 1
                    improved = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()

            # Try 1-swap: remove one node i, add node v if it becomes conflict-free
            in_set_indices = np.where(in_set)[0]
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break
                # Temporarily remove i
                in_set[i] = False
                current_size -= 1
                count_conflicts[neighbors[i]] -= 1

                # Find nodes that can now be added (0 conflicts)
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) > 0:
                    counts = count_conflicts[not_in_set]
                    addable = not_in_set[counts == 0]
                    if len(addable) > 0:
                        # Add the best one (random choice among addable)
                        v = addable[rng.integers(len(addable))]
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                    else:
                        # No addable, restore i
                        in_set[i] = True
                        current_size += 1
                        count_conflicts[neighbors[i]] += 1
                else:
                    # No not_in_set, restore i
                    in_set[i] = True
                    current_size += 1
                    count_conflicts[neighbors[i]] += 1

        return local_best_in_set, local_best_size

    # Multiple restarts
    n_restarts = 0
    max_restarts = 100

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break

        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # 3. Exact polish pass
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 4 and sa_size > current_size:
            polish_time = min(remaining_time * 0.4, 8.0)
            if polish_time > 2:
                polish_result, polish_size = exact_polish(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

        # 4. If we have spare time, do another SA pass on the best
        remaining_time = deadline - 3 - time.time()
        if remaining_time > 5:
            sa2_time = min(remaining_time * 0.6, 10.0)
            if sa2_time > 2:
                sa2_result, sa2_size = sa_local_search(best_in_set, max_time=sa2_time)
                if sa2_size > best_size:
                    best_size = sa2_size
                    best_in_set = sa2_result.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
