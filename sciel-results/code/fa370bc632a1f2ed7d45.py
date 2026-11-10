import numpy as np
import time
import random
import sys

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    # 630s limit, use 590s to be safe
    deadline = start_time + 590
    
    best_set_global = []
    best_size_global = 0
    
    # R values to try
    # R=3 and R=4 are the most promising for high density
    R_values = [3, 4]
    
    for R in R_values:
        if time.time() > deadline - 30:
            break
            
        # Generate candidates with norm_sq <= R
        candidates = []
        
        # Norm 1: e_i, -e_i
        for i in range(d):
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            candidates.append(v)
            candidates.append(-v)
            
        # Norm 2: e_i +/- e_j
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        candidates.append(v)
                        
        # Norm 3: e_i +/- e_j +/- e_k
        if R >= 3:
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                for s3 in [1, -1]:
                                    v = np.zeros(d, dtype=np.int64)
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    candidates.append(v)
                                    
        # Norm 4
        if R >= 4:
            # Type 1: 2e_i
            for i in range(d):
                v = np.zeros(d, dtype=np.int64)
                v[i] = 2
                candidates.append(v)
                candidates.append(-v)
            # Type 2: e_i +/- e_j +/- e_k +/- e_l
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for l in range(k + 1, d):
                            for s1 in [1, -1]:
                                for s2 in [1, -1]:
                                    for s3 in [1, -1]:
                                        for s4 in [1, -1]:
                                            v = np.zeros(d, dtype=np.int64)
                                            v[i] = s1
                                            v[j] = s2
                                            v[k] = s3
                                            v[l] = s4
                                            candidates.append(v)
                                            
        # Unique candidates
        unique_candidates = []
        seen = set()
        for v in candidates:
            if np.all(v == 0):
                continue
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(v)
                
        if len(unique_candidates) < 2:
            continue

        cand_array = np.array(unique_candidates, dtype=np.int64)
        norms_sq = np.sum(cand_array**2, axis=1)
        
        # Filter for current R
        valid_mask = norms_sq <= R
        sub_cands = cand_array[valid_mask]
        sub_norms = norms_sq[valid_mask]
        n_sub = len(sub_cands)
        
        if n_sub < 2:
            continue
            
        # Compute distance matrix
        dots = sub_cands @ sub_cands.T
        dist_sq = sub_norms[:, None] + sub_norms[None, :] - 2 * dots
        
        # Conflict if dist_sq < R
        conflict_mask = dist_sq < R
        np.fill_diagonal(conflict_mask, False)
        
        deg = conflict_mask.sum(axis=1)
        
        # Local Search Strategy:
        # 1. Greedy MIS
        # 2. Remove and Refill
        
        best_set_local = np.zeros(n_sub, dtype=bool)
        best_size_local = 0
        
        # Number of trials
        # R=3 has ~1500 nodes, R=4 has ~7000 nodes
        # Greedy is fast. Local search is O(|S| * N).
        # Let's do 50 trials for R=3 and 10 for R=4?
        # Actually, let's just run until time runs out, but with a cap per R.
        max_trials_R3 = 200
        max_trials_R4 = 50
        max_trials = max_trials_R3 if R == 3 else max_trials_R4
        
        # Pre-sort indices by degree for greedy
        sorted_indices = np.argsort(deg)
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            # 1. Greedy MIS
            # We can randomize the order slightly to escape local optima
            # But min-degree greedy is usually strong.
            # Let's try a mix: 50% min-degree, 50% random order weighted by 1/deg?
            # Simpler: Just use min-degree, but if we get stuck, we rely on local search.
            
            available = np.ones(n_sub, dtype=bool)
            selected = np.zeros(n_sub, dtype=bool)
            
            # Greedy Pick
            # To speed up, we can iterate through sorted_indices
            # But if we pick a node, we must mark its neighbors as unavailable.
            # The "first available in sorted_indices" approach is O(N) per pick if done naively.
            # With N=7000 and ~300 picks, that's 2.1M ops. Fast.
            
            for idx in sorted_indices:
                if available[idx]:
                    selected[idx] = True
                    available[idx] = False
                    # Mark neighbors as unavailable
                    neighbors = np.where(conflict_mask[idx])[0]
                    available[neighbors] = False
                    
            current_size = np.sum(selected)
            
            # 2. Local Search: Remove and Refill
            # We try to improve the set by removing one node and greedily adding back nodes.
            # We repeat this until no improvement is found.
            
            improved = True
            while improved:
                improved = False
                # Get list of selected indices
                sel_indices = np.where(selected)[0]
                
                # Shuffle to avoid bias
                random.shuffle(sel_indices)
                
                for u in sel_indices:
                    if time.time() > deadline - 2:
                        break
                        
                    # Remove u
                    selected[u] = False
                    # Current size
                    size_after_remove = current_size - 1
                    
                    # Find nodes that can be added
                    # A node v can be added if:
                    # 1. v is not in selected
                    # 2. v is not a neighbor of any node in selected
                    # 
                    # Let's compute the "blocked" set.
                    # blocked = union of neighbors of all selected nodes
                    # We can compute this by OR-ing the conflict rows of selected nodes.
                    
                    # Optimization: 
                    # Instead of recomputing from scratch, we can update.
                    # But recomputing is O(|S| * N/word_size) with bitsets, or O(|S| * N) with bool arrays.
                    # |S| ~ 300, N ~ 7000. 2.1M ops. 
                    # Doing this for each u in S (300 times) is 630M ops per pass.
                    # This might be too slow in pure Python/NumPy loop if not vectorized well.
                    
                    # Vectorized approach for "can add":
                    # selected_mask = selected (bool array of length N)
                    # We want to find v such that not selected[v] and not any(conflict_mask[v, selected])
                    # conflict_mask[v, selected] is a boolean vector.
                    # any() is slow.
                    # Sum(conflict_mask[v, selected]) == 0 is better.
                    # Let M = conflict_mask.
                    # We want M[:, selected].sum(axis=1) == 0.
                    # This is a matrix-vector product (bool @ bool).
                    # Or just M[:, selected].any(axis=1).
                    
                    # Let's try:
                    # conflicts_with_selected = M[:, selected].any(axis=1)
                    # This is O(N * |S|).
                    
                    # If we do this for each u, it's O(|S| * N * |S|) = O(N * |S|^2).
                    # 7000 * 90000 = 630M. Too slow for Python.
                    
                    # We need a faster local search.
                    # Alternative: Only try removing a few nodes?
                    # Or, do a single pass of "try to add" after removing u, but limit the candidates?
                    
                    # Let's try a simpler local search:
                    # 1. Identify "bad" nodes (high degree within the set? or just random).
                    # 2. Remove one.
                    # 3. Greedily add nodes that are NOT in the set and NOT conflicting with the set.
                    #    To make this fast, we can just iterate through all nodes and check.
                    #    Check: if not selected[v] and not M[v, selected].any(): add v.
                    #    This is O(N * |S|) per refill.
                    #    If we do this for |S| removals, it's O(N * |S|^2).
                    
                    # Is there a faster way?
                    # Maybe just do a few random removals per pass?
                    
                    # Let's try:
                    # For each u in a random subset of selected nodes (e.g., 10 nodes):
                    #   Remove u
                    #   Greedily refill
                    #   If improved, accept and break
                    # Else, restore u and continue
                    
                    # This reduces the cost per pass.
                    
                    # Let's implement this "limited refill"
                    # We already removed u.
                    # Now refill.
                    
                    # Fast Refill:
                    # Iterate through all indices.
                    # If not selected and not conflicting with current selected, select it.
                    # To check conflict: M[v, selected].any()
                    
                    # We can vectorize the refill step:
                    # candidate_mask = ~selected
                    # conflict_mask_with_selected = M[:, selected].any(axis=1)
                    # add_mask = candidate_mask & ~conflict_mask_with_selected
                    # 
                    # But we must add them one by one? No, if we add a set of nodes that are mutually non-conflicting?
                    # No, the greedy refill should be sequential to maximize size?
                    # Actually, if we add a maximal independent set of the "available" nodes, it's better.
                    # But finding MIS of the available subgraph is the original problem.
                    # So we just do a greedy pass on the "available" nodes.
                    
                    # Greedy pass on available nodes:
                    # available_for_add = add_mask
                    # Sort indices of available_for_add by degree?
                    # Or just random?
                    
                    # Let's just do a simple greedy add:
                    # Iterate through indices.
                    # If add_mask[idx] is True:
                    #    selected[idx] = True
                    #    # Update add_mask for neighbors? 
                    #    # This is expensive if done in a loop.
                    
                    # Let's stick to a simpler heuristic for local search:
                    # Swap: Remove u, add v if possible.
                    # This is O(N) per swap.
                    # Try all u, and for each u, find the best v?
                    
                    # Given the time limit, let's just rely on the Greedy MIS with multiple restarts.
                    # The "remove and refill" might be too slow to implement efficiently in Python without C extensions.
                    # However, the insight says it worked.
                    
                    # Let's try a very fast "Random Restart Greedy" with a twist:
                    # Instead of pure min-degree, we use "Min Degree with Random Tie-Breaking".
                    # And we do many trials.
                    
                    # Let's abandon the complex local search for now and focus on making the Greedy very fast and running many trials.
                    
                    # Restore u if we don't accept the improvement
                    selected[u] = True
                    current_size = size_after_remove + 1
                    break # Break out of u loop if we don't do complex search
                    
            # End Local Search (currently just greedy)
            
            if current_size > best_size_local:
                best_size_local = current_size
                best_set_local = selected.copy()
                
        if best_size_local > best_size_global:
            best_size_global = best_size_local
            best_set_global = [sub_cands[i] for i in np.where(best_set_local)[0]]
            
    # Verify and return
    if best_size_global >= 2:
        result = np.array(best_set_global, dtype=np.int64)
        # Final verification
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = np.dot(result[i] - result[j], result[i] - result[j])
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return result
        
    # Fallback
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Size: {len(res)}")
