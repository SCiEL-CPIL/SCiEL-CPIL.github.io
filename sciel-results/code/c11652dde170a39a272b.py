import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

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

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute conflict matrix: conflict if dot > 2
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    def greedy_min_degree(seed=42):
        """Greedy independent set with min-degree heuristic."""
        rng = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 10:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]

            if len(cands) > 1:
                # Tie-break: prefer vertex whose neighbors have lower average degree
                scores = np.array([
                    np.mean(degrees[neighbors[c]]) if len(neighbors[c]) > 0 else 0
                    for c in cands
                ])
                chosen = cands[np.argmin(scores)]
            else:
                chosen = cands[0]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def local_search(in_set_init, max_time=8.0, seed=42):
        """Simulated annealing local search with add/remove/swap moves."""
        rng = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 5.0
        cooling = 0.9995
        min_T = 0.05

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline and time.time() < deadline - 5:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            move_type = rng.random()

            if move_type < 0.5 and len(not_in_set) > 0:
                # ADD move: try to add a vertex, removing any conflicts
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]

                k = min_count
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
                else:
                    accept = rng.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1

                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()

            elif move_type < 0.7 and len(in_set_indices) > 0:
                # REMOVE move: remove a random vertex to escape local optima
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)

                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1

            elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move: remove one in-set vertex, add one not-in-set vertex
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                v_add_candidates = not_in_set[
                    (count_conflicts[not_in_set] <= 1)
                ]
                if len(v_add_candidates) == 0:
                    v_add_candidates = not_in_set
                v_add = v_add_candidates[rng.integers(len(v_add_candidates))]

                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]

                if len(other_conflicts) == 0:
                    # Clean swap
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                else:
                    # Complex swap: remove v_remove and all other conflicts, add v_add
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1

                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()

            elif len(in_set_indices) > 1 and len(not_in_set) > 1:
                # 2-SWAP move: remove two in-set vertices, add two not-in-set vertices
                v_remove1 = in_set_indices[rng.integers(len(in_set_indices))]
                v_remove2 = in_set_indices[rng.integers(len(in_set_indices))]
                if v_remove1 == v_remove2:
                    continue

                v_add_candidates = not_in_set[
                    (count_conflicts[not_in_set] <= 2)
                ]
                if len(v_add_candidates) < 2:
                    v_add_candidates = not_in_set
                if len(v_add_candidates) < 2:
                    continue
                idx1 = rng.integers(len(v_add_candidates))
                idx2 = rng.integers(len(v_add_candidates))
                if idx1 == idx2:
                    continue
                v_add1 = v_add_candidates[idx1]
                v_add2 = v_add_candidates[idx2]

                # Check if adding both creates conflicts with remaining in_set
                conflicts_1 = np.where(conflicts[v_add1] & in_set)[0]
                conflicts_2 = np.where(conflicts[v_add2] & in_set)[0]
                all_conflicts = np.unique(np.concatenate([conflicts_1, conflicts_2]))
                other_conflicts = all_conflicts[
                    (all_conflicts != v_remove1) & (all_conflicts != v_remove2)
                ]

                if len(other_conflicts) == 0 and v_add1 != v_add2:
                    # Check if v_add1 and v_add2 conflict with each other
                    if not conflicts[v_add1, v_add2]:
                        in_set[v_remove1] = False
                        in_set[v_remove2] = False
                        in_set[v_add1] = True
                        in_set[v_add2] = True
                        count_conflicts[neighbors[v_remove1]] -= 1
                        count_conflicts[neighbors[v_remove2]] -= 1
                        count_conflicts[neighbors[v_add1]] += 1
                        count_conflicts[neighbors[v_add2]] += 1

            T = max(min_T, T * cooling)

        return local_best_in_set, local_best_size

    # Run multiple restarts with different seeds
    num_restarts = 50
    rng = np.random.default_rng(12345)

    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break

        seed = rng.integers(0, 2**31)
        in_set = greedy_min_degree(seed)
        size = int(in_set.sum())

        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()

        # Run local search
        ls_in_set, ls_size = local_search(in_set, max_time=6.0, seed=seed + 1)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_in_set.copy()

    # Final local search on the best solution found
    if best_size > 0 and time.time() < deadline - 10:
        final_in_set, final_size = local_search(best_in_set, max_time=8.0, seed=99999)
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_in_set.copy()

    # Convert best_in_set indices to vectors
    if best_size > 0:
        result = all_candidates[best_in_set]
    else:
        # Fallback to R^2=2 set if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        result = np.array(vectors, dtype=np.int64)

    return result
