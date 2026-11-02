import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1, 0, 1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2, 0, 2}, norm^2=4)
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

    # Precompute neighbor lists as arrays for fast SA updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

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
                # For each remaining node, subtract number of removed neighbors
                # Use conflict matrix slice for efficiency
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict SA with add, remove, and swap moves."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        T = 3.0
        cooling = 0.9997
        min_T = 0.03

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        in_set_indices = np.where(in_set)[0]

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            # Decide move type: add (60%), swap (30%), remove (10%)
            r = rng.random()
            if r < 0.60 and len(not_in_set) > 0:
                # ADD move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
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
                    nbrs_v = neighbors[v]
                    to_remove = nbrs_v[in_set[nbrs_v]]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[nbrs_v] += 1

                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            elif r < 0.90 and len(not_in_set) > 0 and len(in_set_indices) > 0:
                # SWAP move: remove u from set, add v
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]

                # Pick a random u from in_set that is a neighbor of v (to maximize benefit)
                nbrs_v = neighbors[v]
                in_nbrs = nbrs_v[in_set[nbrs_v]]
                if len(in_nbrs) > 0:
                    u = in_nbrs[rng.integers(len(in_nbrs))]
                else:
                    u = in_set_indices[rng.integers(len(in_set_indices))]

                # Compute delta for swap: remove u, add v
                # Removing u: size -1, but may free up neighbors
                # Adding v: size +1, but must remove its neighbors
                # Net: remove u and its effect, add v and remove v's remaining neighbors
                # Simpler: just compute the actual change
                # Remove u
                in_set[u] = False
                count_conflicts[neighbors[u]] -= 1
                size_after_remove = current_size - 1

                # Add v (remove its neighbors first)
                nbrs_v = neighbors[v]
                to_remove = nbrs_v[in_set[nbrs_v]]
                for x in to_remove:
                    in_set[x] = False
                    count_conflicts[neighbors[x]] -= 1
                in_set[v] = True
                count_conflicts[nbrs_v] += 1
                size_after_swap = in_set.sum()
                delta_swap = size_after_swap - current_size

                if delta_swap > 0:
                    accept = True
                elif delta_swap == 0:
                    accept = rng.random() < 0.5
                else:
                    if delta_swap < -20 * T:
                        accept = False
                    else:
                        accept = rng.random() < np.exp(delta_swap / T)

                if not accept:
                    # Undo swap
                    in_set[v] = False
                    count_conflicts[nbrs_v] -= 1
                    for x in to_remove:
                        in_set[x] = True
                        count_conflicts[neighbors[x]] += 1
                    in_set[u] = True
                    count_conflicts[neighbors[u]] += 1
                    current_size = size_after_swap  # restore
                    current_size = local_best_in_set.sum() if False else current_size
                    # Actually we need to restore current_size properly
                    current_size = in_set.sum()
                    stuck_counter += 1
                else:
                    current_size = size_after_swap
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1

            else:
                # REMOVE move (to escape)
                if len(in_set_indices) > 1:
                    u = in_set_indices[rng.integers(len(in_set_indices))]
                    # Only accept with some probability
                    if rng.random() < 0.3:
                        in_set[u] = False
                        count_conflicts[neighbors[u]] -= 1
                        current_size -= 1
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            if stuck_counter > 800:
                # Perturbation: random restart of some elements
                n_perturb = max(3, len(in_set_indices) // 5)
                if len(in_set_indices) > n_perturb:
                    to_flip = rng.choice(in_set_indices, size=n_perturb, replace=False)
                    for idx in to_flip:
                        if in_set[idx]:
                            in_set[idx] = False
                            count_conflicts[neighbors[idx]] -= 1
                            current_size -= 1
                T = max(T, 1.5)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    # Multiple restarts
    n_restarts = 0
    max_restarts = 50

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break

        sa_time = min(remaining_time * 0.6, 15.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # 3. Polish: second SA pass on best found so far
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5:
            polish_time = min(remaining_time * 0.5, 12.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
                # Third polish if time permits
                remaining_time2 = deadline - 3 - time.time()
                if remaining_time2 > 4 and polish_size > sa_size:
                    polish2_time = min(remaining_time2 * 0.6, 8.0)
                    if polish2_time > 2:
                        polish2_result, polish2_size = sa_local_search(polish_result, max_time=polish2_time)
                        if polish2_size > best_size:
                            best_size = polish2_size
                            best_in_set = polish2_result.copy()

    # Final result: combine selected W4 vectors with all W1 vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
