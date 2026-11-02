"""Improved kissing number d=11: expanded candidate pool + fast greedy + SA local search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate candidate vectors with norm^2 = 4
    # W4: weight 4, entries in {-1,0,1}, norm^2=4
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # W1: weight 1, entries in {-2,0,2}, norm^2=4
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

    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    def greedy_min_degree_fast(seed=42):
        """Fast greedy independent set using a heap."""
        rng = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.int32)
        
        # Initialize heap with (-degree, vertex) for min-degree selection
        heap = [(-int(degrees[i]), i) for i in range(N)]
        heapq.heapify(heap)
        
        while heap and time.time() < deadline - 5:
            neg_deg, v = heapq.heappop(heap)
            
            if not alive[v]:
                continue
            
            # Verify degree is still current
            if -neg_deg != degrees[v]:
                continue
            
            in_set[v] = True
            alive[v] = False
            
            # Remove neighbors from consideration
            nbrs = neighbors[v]
            for u in nbrs:
                if alive[u]:
                    alive[u] = False
                    degrees[u] -= 1
                    # Re-push with updated degree
                    heapq.heappush(heap, (-int(degrees[u]), u))
            
            # Also remove v's neighbors' neighbors that might now have lower degree
            # Actually, we just mark them as dead, so no need
        
        return in_set

    def sa_local_search(in_set_init, max_time=8.0, seed=42):
        """Simulated annealing local search with add/remove/swap/kick moves."""
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

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            move_type = rng.random()

            if move_type < 0.45 and len(not_in_set) > 0:
                # ADD move: pick vertex with fewest conflicts
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

            elif move_type < 0.65 and len(in_set_indices) > 0:
                # REMOVE move
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

            elif move_type < 0.8 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move
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
                        stuck_counter += 1

            elif move_type < 0.9 and len(in_set_indices) >= 2 and len(not_in_set) > 0:
                # 2-for-1 move
                v1 = in_set_indices[rng.integers(len(in_set_indices))]
                remaining = np.setdiff1d(in_set_indices, [v1])
                if len(remaining) == 0:
                    stuck_counter += 1
                    continue
                v2 = remaining[rng.integers(len(remaining))]

                v_add_candidates = not_in_set[
                    (count_conflicts[not_in_set] <= 2)
                ]
                if len(v_add_candidates) == 0:
                    v_add_candidates = not_in_set
                v_add = v_add_candidates[rng.integers(len(v_add_candidates))]

                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]

                if len(other_conflicts) == 0:
                    to_remove = np.array([v1, v2])
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                    in_set[v_add] = True
                    current_size -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # KICK move: remove 3-5 random vertices, then greedily re-add
                if len(in_set_indices) >= 3:
                    n_kick = rng.integers(3, min(6, len(in_set_indices) + 1))
                    kick_indices = rng.choice(in_set_indices, size=min(n_kick, len(in_set_indices)), replace=False)
                    for u in kick_indices:
                        in_set[u] = False
                        current_size -= 1
                        count_conflicts[neighbors[u]] -= 1
                    
                    # Greedily re-add vertices with zero conflicts
                    for _ in range(10):
                        not_in_set_now = np.where(~in_set)[0]
                        if len(not_in_set_now) == 0:
                            break
                        zero_conf = not_in_set_now[count_conflicts[not_in_set_now] == 0]
                        if len(zero_conf) == 0:
                            break
                        v = zero_conf[rng.integers(len(zero_conf))]
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
                    stuck_counter += 1

            # Cooling
            T = max(min_T, T * cooling)

            # Perturbation when stuck
            if stuck_counter > 200:
                n_perturb = rng.integers(3, 7)
                in_set_indices_now = np.where(in_set)[0]
                if len(in_set_indices_now) > 0:
                    for _ in range(min(n_perturb, len(in_set_indices_now))):
                        idx = rng.integers(len(in_set_indices_now))
                        u = in_set_indices_now[idx]
                        in_set[u] = False
                        current_size -= 1
                        count_conflicts[neighbors[u]] -= 1
                        in_set_indices_now = np.where(in_set)[0]
                        if len(in_set_indices_now) == 0:
                            break
                stuck_counter = 0
                T = max(T, 3.0)

        # Final polish: greedily add any vertex with zero conflicts
        while True:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            zero_conf = not_in_set[count_conflicts[not_in_set] == 0]
            if len(zero_conf) == 0:
                break
            v = zero_conf[0]
            in_set[v] = True
            current_size += 1
            count_conflicts[neighbors[v]] += 1
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()

        return local_best_in_set, local_best_size

    def polish_set(in_set):
        """Greedily add all zero-conflict vertices."""
        polish = in_set.copy()
        pc = np.zeros(N, dtype=np.int32)
        for i in np.where(polish)[0]:
            pc[neighbors[i]] += 1

        improved = True
        while improved:
            improved = False
            not_in_set = np.where(~polish)[0]
            if len(not_in_set) == 0:
                break
            zero_conf = not_in_set[pc[not_in_set] == 0]
            for v in zero_conf:
                polish[v] = True
                pc[neighbors[v]] += 1
                improved = True

        return polish

    # Multiple restarts
    n_restarts = 0
    max_restarts = 1000
    seed_base = 42

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        seed = seed_base + n_restarts * 7

        in_set = greedy_min_degree_fast(seed=seed)
        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        remaining_time = deadline - 10 - time.time()
        if remaining_time < 3:
            break

        ls_time = min(remaining_time * 0.35, 6.0)
        if ls_time < 1.5:
            ls_time = 1.5

        ls_result, ls_size = sa_local_search(in_set, max_time=ls_time, seed=seed + 1000)

        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

        # Polish
        if best_size > 0:
            polished = polish_set(best_in_set)
            polish_size = int(polished.sum())
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polished.copy()

        if time.time() > deadline - 12:
            break

    # Final polish
    final_set = polish_set(best_in_set)
    final_indices = np.where(final_set)[0]
    result_vectors = all_candidates[final_indices]

    return result_vectors

# EVOLVE-BLOCK-END
