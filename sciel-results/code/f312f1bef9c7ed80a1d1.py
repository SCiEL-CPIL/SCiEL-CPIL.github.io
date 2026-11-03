<<<<<<< SEARCH
    def local_search_swap2(selected_list):
        """Try 2-for-3 swaps: remove two, add three compatible ones."""
        if time.time() > deadline:
            return selected_list
            
        selected_set = set(selected_list)
        if len(selected_set) < 3:
            return list(selected_set)
            
        improved = True
        max_iters = 20
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            sel_list = list(selected_set)
            rng = np.random.default_rng(iters * 97 + 13)
            rng.shuffle(sel_list)
            
            # Try removing pairs
            for i_idx in range(min(len(sel_list), 30)):
                if time.time() > deadline:
                    break
                    
                i = sel_list[i_idx]
                
                for j_idx in range(i_idx + 1, min(i_idx + 15, len(sel_list))):
                    if time.time() > deadline:
                        break
                        
                    j = sel_list[j_idx]
                    
                    # Find candidates that conflict with i or j but not both
                    neighbors_i = set(np.where(Conflict[i])[0].tolist())
                    neighbors_j = set(np.where(Conflict[j])[0].tolist())
                    
                    # Candidates must conflict with i or j (otherwise they'd already be addable)
                    # and must not conflict with any other selected vertex
                    rest_mask = np.zeros(N, dtype=bool)
                    for s in selected_set:
                        if s != i and s != j:
                            rest_mask[s] = True
                    
                    # Find all non-selected vertices that don't conflict with rest
                    non_selected = np.ones(N, dtype=bool)
                    for s in selected_set:
                        non_selected[s] = False
                    
                    conflict_with_rest = Conflict[:, :].sum(axis=1)
                    # We need vertices that don't conflict with rest_mask
                    rest_conflicts = Conflict[:, rest_mask].sum(axis=1)
                    compatible_with_rest = np.where((rest_conflicts == 0) & non_selected)[0]
                    
                    # Filter to those that conflict with i or j
                    candidate_set = set(compatible_with_rest.tolist())
                    candidates_ij = [c for c in candidate_set if c in neighbors_i or c in neighbors_j]
                    
                    if len(candidates_ij) < 3:
                        continue
                    
                    candidates_arr = np.array(candidates_ij, dtype=np.int64)
                    
                    # Find a compatible triple
                    found = False
                    for p in range(len(candidates_arr)):
                        if found:
                            break
                        a = candidates_arr[p]
                        rest_cands = candidates_arr[p+1:]
                        if len(rest_cands) < 2:
                            continue
                        comp_a = Conflict[a, rest_cands] == 0
                        if comp_a.sum() < 2:
                            continue
                        b_idx = np.where(comp_a)[0]
                        b = rest_cands[b_idx[0]]
                        remaining = rest_cands[b_idx[0]+1:]
                        if len(remaining) < 1:
                            continue
                        comp_b = Conflict[b, remaining] == 0
                        if comp_b.any():
                            c = remaining[np.where(comp_b)[0][0]]
                            selected_set.remove(i)
                            selected_set.remove(j)
                            selected_set.add(int(a))
                            selected_set.add(int(b))
                            selected_set.add(int(c))
                            improved = True
                            found = True
                            break
                    
                    if found:
                        break
                
                if found:
                    break
                    
        return list(selected_set)
=======
    def local_search_swap2(selected_list):
        # Removed complex 2-for-3 swap due to performance and bug risks.
        # Replaced with a simpler random 1-for-2 swap variant or no-op.
        # We rely on local_search_swap for improvements.
        return selected_list
>>>>>>> REPLACE

I will also increase the number of restarts in Phase 1 and Phase 3 to compensate for the removal of `swap2`, as `swap1` is faster and can be run more often.

