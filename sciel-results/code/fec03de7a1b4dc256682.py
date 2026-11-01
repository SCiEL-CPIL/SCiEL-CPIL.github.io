"""Kissing number in dimension 11 - Combined Spectral Init + Enhanced SA Local Search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

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

    # Compute conflict matrix: conflict if dot > 2 (since all norms are 2, threshold is ||u||||v||/2 = 4/2 = 2)
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    def spectral_init():
        """
        Spectral initialization:
        1. Compute principal eigenvector of conflict adjacency matrix.
        2. Rank vertices by absolute component value.
        3. Greedily select vertices in this order, skipping conflicts.
        """
        A = conflicts.astype(np.float64)
        
        try:
            from scipy.sparse.linalg import eigsh
            import scipy.sparse as sp
            
            A_sparse = sp.csr_matrix(A)
            vals, vecs = eigsh(A_sparse, k=20, which='LM', return_eigenvectors=True)
            
            idx_max = np.argmax(vals)
            v_main = vecs[:, idx_max]
            scores = np.abs(v_main)
            
        except Exception:
            vals, vecs = np.linalg.eigh(A)
            v_main = vecs[:, -1]
            scores = np.abs(v_main)
        
        order = np.argsort(-scores)
        
        in_set = np.zeros(N, dtype=bool)
        count = 0
        for v in order:
            if not in_set[v] and not np.any(in_set[neighbors[v]]):
                in_set[v] = True
                count += 1
            if time.time() > deadline - 10:
                break
                
        return in_set

    def greedy_min_degree(seed=42):
        """Greedy independent set with min-degree heuristic."""
        rng = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]

            if len(cands) > 1:
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

    def sa_local_search(in_set_init, max_time=8.0, seed=42):
        """Simulated annealing local search with add/remove/swap/2-for-1 moves."""
        rng = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 5.0
        cooling = 0.9995
        min_T = 0.05

        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        step = 0

        while time.time() < sa_deadline:
            step += 1
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            # Try to add a vertex with 0 conflicts first (greedy improvement)
            if len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                zero_conf = not_in_set[counts == 0]
                if len(zero_conf) > 0:
                    v = zero_conf[rng.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    count_conflicts[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    continue

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
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.7 and len(in_set_indices) > 0:
                # REMOVE move: remove a random vertex to escape local optima
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)

                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                    stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move: remove one in-set vertex, add one not-in-set vertex
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                v_add_candidates = not_in_set[(count_conflicts[not_in_set] <= 1)]
                if len(v_add_candidates) == 0:
                    v_add_candidates = not_in_set
                v_add = v_add_candidates[rng.integers(len(v_add_candidates))]

                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]

                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    stuck_counter += 1
                else:
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
                            stuck_counter = 0
                        else:
                            stuck_counter += 1

            else:
                # 2-for-1 grouped swap: find 2 non-conflicting vertices in not_in_set
                # that together conflict with only 1 vertex in in_set
                if stuck_counter > 10:
                    counts = count_conflicts[not_in_set]
                    one_conf = not_in_set[counts == 1]
                    if len(one_conf) > 0:
                        groups = {}
                        for v in one_conf:
                            nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                            if len(nbrs_in_set) == 1:
                                u = nbrs_in_set[0]
                                if u not in groups:
                                    groups[u] = []
                                groups[u].append(v)
                        
                        swap_found = False
                        for u, vs in groups.items():
                            if len(vs) >= 2:
                                for i in range(len(vs)):
                                    for j in range(i+1, len(vs)):
                                        v1, v2 = vs[i], vs[j]
                                        if not conflicts[v1, v2]:
                                            in_set[u] = False
                                            in_set[v1] = True
                                            in_set[v2] = True
                                            
                                            count_conflicts[neighbors[u]] -= 1
                                            count_conflicts[neighbors[v1]] += 1
                                            count_conflicts[neighbors[v2]] += 1
                                            
                                            current_size += 1
                                            swap_found = True
                                            
                                            if current_size > local_best_size:
                                                local_best_size = current_size
                                                local_best_in_set = in_set.copy()
                                            break
                                    if swap_found:
                                        break
                                if swap_found:
                                    break
                        if swap_found:
                            stuck_counter = 0
                            continue

            if stuck_counter > 20:
                # Random perturbation
                if len(in_set_indices) > 0:
                    u = in_set_indices[rng.integers(len(in_set_indices))]
                    in_set[u] = False
                    current_size -= 1
                    count_conflicts[neighbors[u]] -= 1
                    stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    # 1. Spectral Init
    in_set = spectral_init()
    size = in_set.sum()
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
        
    # Local Search on spectral init
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 2:
        ls_time = min(remaining_time * 0.4, 15.0)
        ls_result, ls_size = sa_local_search(best_in_set, max_time=ls_time, seed=42)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
    # 2. Greedy Init (multiple restarts)
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 20:
        n_restarts += 1
        in_set = greedy_min_degree(seed=42 + n_restarts)
        size = in_set.sum()
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
            
        ls_time = min(remaining_time * 0.3, 8.0)
        ls_result, ls_size = sa_local_search(in_set, max_time=ls_time, seed=100 + n_restarts)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # Final polish: try to add any remaining vertices with 0 conflicts
    if time.time() < deadline - 2:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        not_in_set = np.where(~best_in_set)[0]
        if len(not_in_set) > 0:
            counts = conflict_count[not_in_set]
            zero_conf = not_in_set[counts == 0]
            for v in zero_conf:
                best_in_set[v] = True
                conflict_count[neighbors[v]] += 1
                remaining = np.where(~best_in_set)[0]
                if len(remaining) > 0:
                    conflict_count[remaining] += conflicts[remaining, v].astype(np.int32)
    
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors

# EVOLVE-BLOCK-END
