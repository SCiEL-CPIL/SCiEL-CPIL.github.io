import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Generate all W4 vectors: weight-4, entries in {-1,0,1}, norm^2=4
    v4_candidates = []
    all_subsets = list(itertools.combinations(range(D), 4))
    for positions in all_subsets:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors: weight-1, entries in {-2,0,2}, norm^2=4
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(v4_candidates)

    if N4 == 0:
        return vectors_v1

    # Compute conflict matrix: conflict if dot > 2
    # For W4 vectors, dot product is sum of products of overlapping entries
    # Two W4 vectors conflict if |dot| > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = np.abs(dots) > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists as arrays
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
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing local search."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

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

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break

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

    def branch_and_bound_max_clique(compat_matrix, time_limit=5.0):
        """
        Branch-and-bound for maximum clique in compatibility graph.
        compat_matrix: boolean matrix where True means compatible (non-conflicting)
        Returns the largest clique found within time_limit.
        """
        n = compat_matrix.shape[0]
        if n == 0:
            return np.zeros(n, dtype=bool)

        # Convert to adjacency list for faster access
        adj = [np.where(compat_matrix[i])[0] for i in range(n)]

        best_clique = np.zeros(n, dtype=bool)
        best_size = 0

        # Greedy coloring upper bound for a set of vertices
        def color_bound(vertices):
            """Greedy coloring to get an upper bound on clique size."""
            if len(vertices) == 0:
                return 0
            colors = {}
            color_count = 0
            remaining = list(vertices)
            while remaining:
                color_count += 1
                color_set = set()
                new_remaining = []
                for v in remaining:
                    if all(v not in color_set or not compat_matrix[v, u] for u in color_set):
                        color_set.add(v)
                        colors[v] = color_count
                    else:
                        new_remaining.append(v)
                remaining = new_remaining
            return color_count

        def bbsolve(candidates, current_clique, current_size):
            nonlocal best_clique, best_size

            if time.time() > deadline - 3:
                return

            if len(candidates) == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_clique = current_clique.copy()
                return

            # Upper bound: current_size + len(candidates)
            if current_size + len(candidates) <= best_size:
                return

            # Use coloring bound for tighter upper bound
            color_ub = color_bound(candidates)
            if current_size + color_ub <= best_size:
                return

            # Order candidates by degree in subgraph (descending)
            cand_list = list(candidates)
            degs = np.array([np.sum(compat_matrix[v, candidates]) for v in cand_list])
            order = np.argsort(-degs)
            cand_list = [cand_list[i] for i in order]

            for i, v in enumerate(cand_list):
                if time.time() > deadline - 3:
                    return

                # Try adding v to clique
                new_candidates = [u for u in cand_list[i+1:] if compat_matrix[v, u]]
                new_clique = current_clique.copy()
                new_clique[v] = True
                bbsolve(new_candidates, new_clique, current_size + 1)

                # Remove v from consideration
                # (implicit in the loop structure)

        # Initialize with greedy clique
        greedy_clique = np.zeros(n, dtype=bool)
        remaining = np.ones(n, dtype=bool)
        while np.any(remaining):
            # Pick vertex with max degree in remaining subgraph
            degs = np.array([np.sum(compat_matrix[i, remaining]) for i in np.where(remaining)[0]])
            if len(degs) == 0:
                break
            max_idx = np.where(remaining)[0][np.argmax(degs)]
            greedy_clique[max_idx] = True
            remaining[max_idx] = False
            remaining[~compat_matrix[max_idx]] = False

        if greedy_clique.sum() > best_size:
            best_size = greedy_clique.sum()
            best_clique = greedy_clique.copy()

        # Run B&B
        bbsolve(list(range(n)), np.zeros(n, dtype=bool), 0)

        return best_clique

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

        sa_time = min(remaining_time * 0.5, 15.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # 3. B&B on a subset of candidates for refinement
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 10 and sa_size > current_size:
            # Focus on vertices near the current solution
            # Take the current solution plus some nearby candidates
            current_indices = np.where(best_in_set)[0]
            # Add neighbors of current solution as candidates
            candidate_set = set(current_indices.tolist())
            for idx in current_indices:
                candidate_set.update(neighbors[idx].tolist())
            candidate_list = list(candidate_set)
            
            if 10 < len(candidate_list) < 500:
                # Build sub-compatibility matrix
                idx_map = {v: i for i, v in enumerate(candidate_list)}
                sub_compat = np.zeros((len(candidate_list), len(candidate_list)), dtype=bool)
                for i, vi in enumerate(candidate_list):
                    for j, vj in enumerate(candidate_list):
                        if i != j:
                            sub_compat[i, j] = not conflicts[vi, vj]
                
                bb_time = min(remaining_time * 0.3, 8.0)
                if bb_time > 2:
                    bb_result = branch_and_bound_max_clique(sub_compat, time_limit=bb_time)
                    bb_indices = [candidate_list[i] for i in np.where(bb_result)[0]]
                    if len(bb_indices) > best_size:
                        new_in_set = np.zeros(N4, dtype=bool)
                        new_in_set[bb_indices] = True
                        best_size = len(bb_indices)
                        best_in_set = new_in_set

        # 4. Second SA pass on best found so far (polish)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.4, 10.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(best_in_set, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors
