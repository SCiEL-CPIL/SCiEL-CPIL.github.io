"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin before 150s kill

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
    
    # Compatibility matrix (for clique formulation): edge if NOT conflicting
    compat = ~conflicts
    np.fill_diagonal(compat, True)
    
    # Precompute neighbor lists for independent set formulation
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Compute "expansion" score: number of edges among neighbors (lower = better to pick first)
    expansion_scores = np.zeros(N4, dtype=np.int32)
    for i in range(N4):
        nbrs = neighbors[i]
        if len(nbrs) > 1:
            sub_conf = conflicts[np.ix_(nbrs, nbrs)]
            expansion_scores[i] = sub_conf.sum() // 2
    
    rng = np.random.default_rng(12345)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_weighted_init(seed=None):
        """Greedy independent set with expansion-aware heuristic."""
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = rng
            
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Score: primarily degree, secondarily expansion
            degs = degrees[alive_indices]
            mins = np.min(degs)
            cands = alive_indices[degs == mins]
            
            if len(cands) > 1:
                exps = expansion_scores[cands]
                min_exp = np.min(exps)
                final_cands = cands[exps == min_exp]
                chosen = final_cands[local_rng.integers(len(final_cands))]
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
    
    def sa_local_search(in_set_init, max_time=8.0, T_start=2.0):
        """Simulated annealing with add/remove/swap/2-swap moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = T_start
        cooling = 0.9997
        min_T = 0.02
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        step = 0
        
        while time.time() < sa_deadline:
            step += 1
            in_set_indices = np.where(in_set)[0]
            not_in_set = np.where(~in_set)[0]
            
            if len(not_in_set) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.55 and len(not_in_set) > 0:
                # ADD move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                
                if len(min_cands) > 1:
                    degs = neighbor_counts[min_cands]
                    min_deg = np.min(degs)
                    v = min_cands[degs == min_deg][rng.integers(np.sum(degs == min_deg))]
                else:
                    v = min_cands[0]
                
                k = int(min_count)
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
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
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.70 and len(in_set_indices) > 0:
                # REMOVE move
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                n_sample = min(15, len(not_in_set))
                sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                sample_cands = not_in_set[sample_idx]
                
                temp_counts = count_conflicts.copy()
                temp_counts[neighbors[v_remove]] -= 1
                temp_counts[v_remove] = 0
                
                cands_conf = temp_counts[sample_cands]
                min_conf = np.min(cands_conf)
                good_cands = sample_cands[cands_conf == min_conf]
                
                if len(good_cands) > 0:
                    v_add = good_cands[rng.integers(len(good_cands))]
                    delta = 1 - int(min_conf)
                    
                    if delta >= 0 or rng.random() < np.exp(delta / T):
                        to_remove = np.where(conflicts[v_add] & in_set)[0]
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                else:
                    stuck_counter += 1
                    
            else:
                # 2-SWAP
                if len(in_set_indices) >= 2 and len(not_in_set) >= 2:
                    v1 = in_set_indices[rng.integers(len(in_set_indices))]
                    v2 = in_set_indices[rng.integers(len(in_set_indices))]
                    if v1 == v2:
                        v2 = in_set_indices[rng.integers(len(in_set_indices))]
                        if v1 == v2:
                            stuck_counter += 1
                        else:
                            n_sample = min(10, len(not_in_set))
                            sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                            sample_cands = not_in_set[sample_idx]
                            
                            found = []
                            for v_add in sample_cands:
                                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                                other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]
                                if len(other_conflicts) == 0:
                                    found.append(v_add)
                                    if len(found) == 2:
                                        break
                            
                            if len(found) == 2:
                                to_remove = np.array([v1, v2], dtype=np.int64)
                                in_set[to_remove] = False
                                for u in to_remove:
                                    count_conflicts[neighbors[u]] -= 1
                                for v_ad in found:
                                    in_set[v_ad] = True
                                    count_conflicts[neighbors[v_ad]] += 1
                            else:
                                stuck_counter += 1
                    else:
                        n_sample = min(10, len(not_in_set))
                        sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                        sample_cands = not_in_set[sample_idx]
                        
                        found = []
                        for v_add in sample_cands:
                            conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                            other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]
                            if len(other_conflicts) == 0:
                                found.append(v_add)
                                if len(found) == 2:
                                    break
                        
                        if len(found) == 2:
                            to_remove = np.array([v1, v2], dtype=np.int64)
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                            for v_ad in found:
                                in_set[v_ad] = True
                                count_conflicts[neighbors[v_ad]] += 1
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 1500:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    def greedy_polish(in_set_init):
        """Fast greedy: add any vertex that doesn't conflict with current set."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            
            addable = not_in[count_conflicts[not_in] == 0]
            
            if len(addable) > 0:
                v_idx = addable[0]
                in_set[v_idx] = True
                current_size += 1
                count_conflicts[neighbors[v_idx]] += 1
                changed = True
        
        return in_set, current_size
    
    # Maximum Clique via Branch and Bound with greedy coloring
    def max_clique_bnb(vertices, max_time=5.0):
        """
        Find maximum clique in compatibility graph induced by vertices.
        Uses branch and bound with greedy coloring upper bound.
        Returns the clique as a set of vertex indices.
        """
        if len(vertices) == 0:
            return set()
        
        vertices = list(vertices)
        n = len(vertices)
        if n == 0:
            return set()
        
        # Build local adjacency for these vertices
        # local_adj[i] = set of local indices j (j != i) such that compat[vertices[i], vertices[j]]
        local_adj = [set() for _ in range(n)]
        for i in range(n):
            vi = vertices[i]
            for j in range(i+1, n):
                vj = vertices[j]
                if compat[vi, vj]:
                    local_adj[i].add(j)
                    local_adj[j].add(i)
        
        # Greedy coloring for upper bound
        def greedy_color(order):
            """
            Greedy coloring of the subgraph induced by vertices in 'order'.
            Returns (coloring, max_color) where coloring[i] is color of vertex at position i in order.
            The color of a vertex is an upper bound on clique size in the subgraph from that vertex onward.
            """
            coloring = [0] * len(order)
            color_class = [set() for _ in range(len(order))]
            max_color = 0
            
            for idx, v in enumerate(order):
                # Find smallest color not used by neighbors of v that come after v in order
                # Actually, standard greedy coloring for upper bound:
                # Process vertices in reverse order, assign smallest available color
                pass
            
            # Simpler approach: process in given order, assign smallest available color
            used_colors = [set() for _ in range(len(order))]
            for idx in range(len(order)):
                v = order[idx]
                # Colors used by neighbors of v that are already colored (earlier in order)
                forbidden = set()
                for nb in local_adj[v]:
                    if nb < idx:
                        forbidden.add(coloring[nb])
                
                c = 0
                while c in forbidden:
                    c += 1
                coloring[idx] = c
                if c > max_color:
                    max_color = c
            
            return coloring, max_color
        
        best_clique = set()
        
        def expand(clique_set, candidates, max_color_bound):
            nonlocal best_clique
            
            if time.time() > max_time:
                return
            
            if len(clique_set) > len(best_clique):
                best_clique = set(clique_set)
            
            if not candidates:
                return
            
            # Order candidates by degree in candidate subgraph (descending)
            cand_list = list(candidates)
            
            # Greedy color the candidates for upper bound
            # We'll use a simple sequential coloring
            color_arr = [0] * len(cand_list)
            max_col = 0
            for i in range(len(cand_list)):
                v = cand_list[i]
                forbidden = set()
                for j in range(i):
                    if j in local_adj[v]:
                        forbidden.add(color_arr[j])
                c = 0
                while c in forbidden:
                    c += 1
                color_arr[i] = c
                if c > max_col:
                    max_col = c
            
            # Process in reverse order of coloring (highest color first)
            # This gives tighter bounds
            order = sorted(range(len(cand_list)), key=lambda i: color_arr[i], reverse=True)
            
            for pos in order:
                v = cand_list[pos]
                
                # Upper bound: current clique size + color of v
                if len(clique_set) + color_arr[pos] <= len(best_clique):
                    break
                
                # Branch: include v
                new_clique = clique_set | {v}
                
                # New candidates: neighbors of v in candidates, with index > pos in order
                # Actually, we need candidates that are neighbors of v and come after v in the processing order
                # To avoid duplicates, we only consider candidates that appear after v in our ordering
                new_candidates = set()
                for nb in local_adj[v]:
                    if nb in candidates and cand_list.index(nb) > pos:
                        new_candidates.add(nb)
                
                # Recurse
                expand(new_clique, new_candidates, max_color_bound)
                
                # Remove v from candidates for subsequent branches
                candidates.discard(v)
        
        # Initial call
        all_candidates = set(range(n))
        expand(set(), all_candidates, n)
        
        # Map back to original vertex indices
        result = set()
        for local_idx in best_clique:
            result.add(vertices[local_idx])
        
        return result
    
    def greedy_clique_init(seed=None):
        """Greedy maximum clique heuristic on compatibility graph."""
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = rng
        
        all_vertices = set(range(N4))
        best_clique = set()
        
        # Try multiple random orderings
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random order
            order = list(range(N4))
            local_rng.shuffle(order)
            
            # Greedy clique construction
            clique = set()
            candidates = set(order)
            
            for v in order:
                if v not in candidates:
                    continue
                # Check if v is compatible with all in clique
                compatible = True
                for u in clique:
                    if not compat[v, u]:
                        compatible = False
                        break
                if compatible:
                    clique.add(v)
                    # Remove non-compatible vertices from candidates
                    to_remove = set()
                    for u in candidates:
                        if u != v and not compat[v, u]:
                            to_remove.add(u)
                    candidates -= to_remove
            
            if len(clique) > len(best_clique):
                best_clique = clique.copy()
        
        return best_clique
    
    def clique_to_independent_set(clique_local):
        """Convert clique in compatibility graph to independent set in conflict graph."""
        in_set = np.zeros(N4, dtype=bool)
        for v in clique_local:
            in_set[v] = True
        return in_set
    
    # Main search loop
    n_restarts = 0
    max_restarts = 150
    luby_pattern = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 8, 16]
    luby_idx = 0
    
    while time.time() < deadline - 6 and n_restarts < max_restarts:
        n_restarts += 1
        
        luby_len = luby_pattern[luby_idx % len(luby_pattern)]
        luby_idx += 1
        
        remaining_time = deadline - 6 - time.time()
        if remaining_time < 3:
            break
        
        # Strategy: alternate between independent set SA and clique B&B
        if n_restarts % 3 == 0:
            # Use clique B&B approach
            seed = n_restarts * 7 + 13
            # Get initial clique via greedy
            greedy_clique = greedy_clique_init(seed=seed)
            
            # Convert to independent set for SA refinement
            in_set = clique_to_independent_set(greedy_clique)
            current_size = int(in_set.sum())
            
            if current_size > best_size:
                best_size = current_size
                best_in_set = in_set.copy()
            
            # SA refinement
            sa_time = min(remaining_time * 0.4, 8.0)
            if sa_time > 1:
                sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
                if sa_size > best_size:
                    best_size = sa_size
                    best_in_set = sa_result.copy()
            
            # Try B&B on the neighborhood of the best clique
            remaining_time = deadline - 4 - time.time()
            if remaining_time > 3 and best_size > 0:
                # Get the vertices in the best clique
                best_clique_vertices = set(np.where(best_in_set)[0].tolist())
                
                # Expand search to include neighbors of the clique
                # For B&B, use a subset of vertices around the current best
                expanded_vertices = set(best_clique_vertices)
                for v in best_clique_vertices:
                    for nb in neighbors[v]:
                        if not compat[v, nb]:  # Only add conflicting neighbors (potential replacements)
                            expanded_vertices.add(nb)
                
                # Limit size for B&B
                if len(expanded_vertices) > 200:
                    # Sample
                    expanded_vertices = set(np.random.choice(list(expanded_vertices), size=200, replace=False))
                
                bnb_time = min(remaining_time * 0.5, 6.0)
                if bnb_time > 1:
                    clique_result = max_clique_bnb(list(expanded_vertices), max_time=bnb_time)
                    if len(clique_result) > best_size:
                        new_in_set = clique_to_independent_set(clique_result)
                        best_size = len(clique_result)
                        best_in_set = new_in_set
        else:
            # Standard independent set approach
            seed = n_restarts * 7 + 13
            in_set = greedy_weighted_init(seed=seed)
            current_size = int(in_set.sum())
            
            if current_size > best_size:
                best_size = current_size
                best_in_set = in_set.copy()
            
            # SA Local Search
            base_sa_time = min(remaining_time * 0.5, 10.0)
            sa_time = base_sa_time * (luby_len / 4.0)
            sa_time = max(sa_time, 1.5)
            sa_time = min(sa_time, remaining_time * 0.7)
            
            sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
            
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()
            
            # Greedy polish
            remaining_time = deadline - 4 - time.time()
            if remaining_time > 2:
                polish_set, polish_size = greedy_polish(sa_result if sa_size >= current_size else in_set)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_set.copy()
            
            # Second SA on best found
            remaining_time = deadline - 3 - time.time()
            if remaining_time > 5 and n_restarts % 3 == 2:
                polish_time = min(remaining_time * 0.6, 8.0)
                if polish_time > 2:
                    polish_result, polish_size = sa_local_search(best_in_set, max_time=polish_time, T_start=1.0)
                    if polish_size > best_size:
                        best_size = polish_size
                        best_in_set = polish_result.copy()
    
    # Final greedy polish on best result
    if time.time() < deadline - 2:
        final_polish, final_size = greedy_polish(best_in_set)
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_polish
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
