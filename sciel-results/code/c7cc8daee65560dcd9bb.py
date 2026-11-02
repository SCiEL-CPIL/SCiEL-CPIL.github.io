"""Combined: min-degree greedy + Tomita MIS + enhanced swap local search for kissing number d=11."""
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s limit

    D = 11

    # Generate all candidates: weight-4 (5280), weight-2 (220), weight-1 (22)
    candidates_list = []
    # Weight-4: C(11,4)*16 = 5280
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    # Weight-2: C(11,2)*4 = 220
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    # Weight-1: 22
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5522

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Precompute degree
    degree = conflict.sum(axis=1).astype(np.int32)

    def greedy_min_degree(remaining_init, time_limit):
        """Min-degree greedy: pick vertex with fewest conflicts among remaining."""
        remaining = remaining_init.copy()
        selected = []
        while time.time() < time_limit and remaining.any():
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & (remaining > 0)
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            selected.append(new_idx)
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0
        return selected

    def tomita_mis_bitset(candidates_pool, initial_selected, time_limit):
        """Tomita-style branch and bound MIS using bitset operations.
        candidates_pool: set of vertices to consider for addition
        initial_selected: current selected set (all compatible with each other)
        Returns improved selected list.
        """
        start_t = time.time()
        
        # Work with the subgraph induced by candidates_pool
        # We want to find a large independent set in the conflict graph
        # that is also compatible with all initial_selected
        
        # Filter: only vertices compatible with all initial_selected
        if initial_selected:
            mask_compat = np.ones(len(candidates_pool), dtype=bool)
            for s in initial_selected:
                mask_compat = mask_compat & (compat[candidates_pool, s] > 0)
            pool = candidates_pool[mask_compat]
        else:
            pool = candidates_pool
        
        if len(pool) == 0:
            return list(initial_selected)
        
        # Build local index mapping
        pool_set = set(pool.tolist())
        local_to_global = {local_idx: pool[local_idx] for local_idx in range(len(pool))}
        
        # Build conflict subgraph as bitsets
        # conflict_local[i] = set of local indices that conflict with pool[i]
        # Use Python sets for simplicity, but optimize with numpy where possible
        
        n_local = len(pool)
        
        # Precompute conflict neighbors for each local vertex
        conflict_neighbors = []
        for i in range(n_local):
            gi = pool[i]
            # Find which local vertices conflict with gi
            cn = set()
            for j in range(n_local):
                if i != j and conflict[gi, pool[j]]:
                    cn.add(j)
            conflict_neighbors.append(cn)
        
        # Greedy coloring for upper bound
        def greedy_color(vertices):
            """Greedy sequential coloring, returns (color_dict, sorted_order)."""
            if not vertices:
                return {}, []
            
            vset = set(vertices)
            color = {}
            
            # Sort by degree in induced subgraph (descending)
            degs = {v: len(conflict_neighbors[v] & vset) for v in vertices}
            order = sorted(vertices, key=lambda v: -degs[v])
            
            for v in order:
                neighbor_colors = set()
                for u in conflict_neighbors[v]:
                    if u in vset and u in color:
                        neighbor_colors.add(color[u])
                c = 1
                while c in neighbor_colors:
                    c += 1
                color[v] = c
            
            # Sort by color for branching
            sorted_order = sorted(vertices, key=lambda v: color[v])
            return color, sorted_order
        
        best_size = len(initial_selected)
        best_solution = list(initial_selected)
        
        # Recursive branch and bound
        def bb(P, Q):
            nonlocal best_size, best_solution
            
            if time.time() - start_t > time_limit:
                return
            
            if not P:
                if len(Q) > best_size:
                    best_size = len(Q)
                    best_solution = Q[:]
                return
            
            # Upper bound via coloring
            color, sorted_order = greedy_color(P)
            if not color:
                if len(Q) > best_size:
                    best_size = len(Q)
                    best_solution = Q[:]
                return
            
            max_color = max(color.values())
            if len(Q) + max_color <= best_size:
                return
            
            # Branch in reverse color order (higher colors first)
            for v in reversed(sorted_order):
                if time.time() - start_t > time_limit:
                    return
                
                # Include v in solution
                Q.append(v)
                
                # New P: vertices in P compatible with v (not conflicting)
                P_new = [u for u in P if u != v and u not in conflict_neighbors[v]]
                
                bb(P_new, Q)
                
                Q.pop()
                
                # Remove v from P for next iteration
                P.remove(v)
                
                # Prune check
                if len(Q) + max_color <= best_size:
                    return
        
        # Initial call: P = all pool vertices, Q = initial_selected
        bb(list(range(n_local)), list(initial_selected))
        
        # Convert back to global indices
        if best_solution:
            result = list(initial_selected)
            for v in best_solution[len(initial_selected):]:
                result.append(local_to_global[v])
            return result
        return list(initial_selected)

    def enhanced_local_search(selected_list, time_limit, max_rounds=500):
        """Enhanced local search with multiple swap strategies."""
        selected = list(selected_list)
        sel_set = set(selected)
        
        # Precompute compat_with_all
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0
        
        full_deg = conflict.sum(axis=1)
        
        for round_num in range(max_rounds):
            if time.time() > time_limit:
                break
            
            # Strategy 1: Simple addition
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                # Pick least constraining
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                continue
            
            # Strategy 2: 1-for-2 swap
            found_swap = False
            trial_count = min(15, len(selected))
            if len(selected) > 30:
                trial_indices = np.random.choice(len(selected), size=trial_count, replace=False)
            else:
                trial_indices = range(len(selected))
            
            for si in trial_indices:
                if time.time() > time_limit:
                    break
                s = selected[si]
                
                # Find candidates compatible with all selected except s
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[:, s2]
                mask_except_s[s] = 0
                for s3 in sel_set:
                    mask_except_s[s3] = 0
                
                cands = np.where(mask_except_s > 0)[0]
                if len(cands) < 2:
                    continue
                
                # Sort by degree
                sorted_cands = cands[np.argsort(full_deg[cands])]
                
                # Find compatible pair
                found_pair = False
                max_check = min(30, len(sorted_cands))
                for i in range(max_check):
                    c1 = sorted_cands[i]
                    for j in range(i + 1, min(i + 20, max_check)):
                        c2 = sorted_cands[j]
                        if compat[c1, c2]:
                            # Do swap
                            selected.remove(s)
                            sel_set.discard(s)
                            selected.append(c1)
                            sel_set.add(c1)
                            selected.append(c2)
                            sel_set.add(c2)
                            # Recompute compat_with_all
                            compat_with_all = np.ones(n, dtype=np.uint8)
                            for s3 in selected:
                                compat_with_all = compat_with_all & compat[s3]
                            for s3 in selected:
                                compat_with_all[s3] = 0
                            found_swap = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_swap:
                    break
            
            if found_swap:
                continue
            
            # Strategy 3: 2-for-3 swap (remove 2, add 3)
            found_2for3 = False
            if len(selected) >= 4:
                # Try removing pairs of selected vertices
                pair_trials = min(8, len(selected) * (len(selected) - 1) // 2)
                pair_list = []
                for a in range(len(selected)):
                    for b in range(a + 1, len(selected)):
                        pair_list.append((a, b))
                        if len(pair_list) >= pair_trials:
                            break
                    if len(pair_list) >= pair_trials:
                        break
                
                for (ia, ib) in pair_list:
                    if time.time() > time_limit:
                        break
                    sa, sb = selected[ia], selected[ib]
                    
                    # Candidates compatible with all selected except sa, sb
                    mask_except = np.ones(n, dtype=np.uint8)
                    for s2 in selected:
                        if s2 != sa and s2 != sb:
                            mask_except = mask_except & compat[:, s2]
                    mask_except[sa] = 0
                    mask_except[sb] = 0
                    for s3 in sel_set:
                        mask_except[s3] = 0
                    
                    cands = np.where(mask_except > 0)[0]
                    if len(cands) < 3:
                        continue
                    
                    sorted_cands = cands[np.argsort(full_deg[cands])]
                    max_check = min(20, len(sorted_cands))
                    
                    # Find 3 mutually compatible
                    found_triple = False
                    for i in range(max_check):
                        c1 = sorted_cands[i]
                        for j in range(i + 1, min(i + 15, max_check)):
                            c2 = sorted_cands[j]
                            if not compat[c1, c2]:
                                continue
                            for k in range(j + 1, min(j + 15, max_check)):
                                c3 = sorted_cands[k]
                                if compat[c1, c3] and compat[c2, c3]:
                                    # Do 2-for-3 swap
                                    selected.remove(sa)
                                    selected.remove(sb)
                                    sel_set.discard(sa)
                                    sel_set.discard(sb)
                                    selected.append(c1)
                                    selected.append(c2)
                                    selected.append(c3)
                                    sel_set.add(c1)
                                    sel_set.add(c2)
                                    sel_set.add(c3)
                                    # Recompute
                                    compat_with_all = np.ones(n, dtype=np.uint8)
                                    for s3 in selected:
                                        compat_with_all = compat_with_all & compat[s3]
                                    for s3 in selected:
                                        compat_with_all[s3] = 0
                                    found_2for3 = True
                                    found_triple = True
                                    break
                            if found_triple:
                                break
                        if found_triple:
                            break
                    if found_2for3:
                        break
            
            if not found_swap and not found_2for3:
                break
        
        return selected

    # Strategy: multiple min-degree greedy restarts
    best_selected = []
    np.random.seed(123)

    # First: pure min-degree greedy on all candidates
    sel = greedy_min_degree(np.ones(n, dtype=np.uint8), deadline - 80.0)
    if len(sel) > len(best_selected):
        best_selected = sel

    # Random restarts with different subsets
    for trial in range(40):
        if time.time() > deadline - 50.0:
            break
        frac = np.random.uniform(0.4, 1.0)
        subset_size = int(n * frac)
        subset_indices = np.random.choice(n, size=subset_size, replace=False)
        remaining_init = np.zeros(n, dtype=np.uint8)
        remaining_init[subset_indices] = 1
        sel = greedy_min_degree(remaining_init, deadline - 50.0)
        if len(sel) > len(best_selected):
            best_selected = sel

    # Try Tomita MIS on a reduced candidate pool
    if best_selected and time.time() < deadline - 20.0:
        # Use Tomita on candidates compatible with current best
        compatible_with_best = np.where(compat_with_all > 0)[0] if len(best_selected) > 0 else np.arange(n)
        if len(compatible_with_best) > 10:
            remaining_time = deadline - time.time() - 10.0
            if remaining_time > 5:
                try:
                    improved = tomita_mis_bitset(compatible_with_best, best_selected, min(remaining_time, 30.0))
                    if len(improved) > len(best_selected):
                        best_selected = improved
                except Exception:
                    pass

    # Enhanced local search
    if best_selected and time.time() < deadline - 10.0:
        improved = enhanced_local_search(best_selected, deadline - 10.0, max_rounds=400)
        if len(improved) > len(best_selected):
            best_selected = improved

    # Final greedy fill
    if best_selected and time.time() < deadline - 3.0:
        sel_set = set(best_selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in best_selected:
            compat_with_all = compat_with_all & compat[s]
        for s in best_selected:
            compat_with_all[s] = 0
        addable = np.where(compat_with_all > 0)[0]
        full_deg = conflict.sum(axis=1)
        for idx in addable[np.argsort(full_deg[addable])]:
            if time.time() > deadline - 3.0:
                break
            best_selected.append(idx)
            sel_set.add(idx)
            compat_with_all = compat_with_all & compat[idx]
            compat_with_all[idx] = 0

    if not best_selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_selected, dtype=np.int64)]
    return result.astype(np.int64)
