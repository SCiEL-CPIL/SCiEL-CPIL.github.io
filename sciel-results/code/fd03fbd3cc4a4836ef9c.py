import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    best_set = []
    best_size = 0
    
    # Helper: Run greedy
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    # Helper: Vectorized Local Search (1-for-2 swaps)
    def local_search(selected_indices):
        if time.time() > deadline:
            return selected_indices
            
        S = set(selected_indices)
        # conf_counts[j] = number of neighbors of j in S
        # We only care about j not in S, but computing for all is fine
        conf_counts = np.zeros(N, dtype=np.int64)
        for i in S:
            conf_counts += Conflict[i]
        
        # Zero out counts for nodes in S (they are not candidates)
        # Actually, we just need to check j not in S.
        # We will maintain a boolean mask in_S
        in_S = np.zeros(N, dtype=bool)
        in_S[list(S)] = True
        
        improved = True
        max_iterations = 500 # Safety cap for local search steps
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            # Shuffle S for randomization
            s_list = list(S)
            np.random.shuffle(s_list)
            
            for i in s_list:
                if time.time() > deadline:
                    return list(S)
                
                # Find candidates j not in S such that conf_counts[j] == Conflict[j, i]
                # This means j conflicts with i and no other node in S
                # If Conflict[j, i] is 0, conf_counts[j] must be 0 (compatible with all S)
                # But greedy ensures no such j exists. So we look for Conflict[j, i] == 1 and conf_counts[j] == 1.
                
                # Vectorized check:
                # candidates_mask = (~in_S) & (conf_counts == Conflict[i])
                # Note: Conflict[i] is a vector of 0s and 1s.
                # We want j where conf_counts[j] == Conflict[i, j]
                
                # Since Conflict is symmetric, Conflict[i, j] == Conflict[j, i]
                # We need conf_counts[j] == Conflict[j, i]
                
                # Let's create a mask of valid candidates
                # valid_j = ~in_S
                # condition = (conf_counts == Conflict[i]) # This compares conf_counts vector with Conflict[i] vector
                # Wait, Conflict[i] is the row i. Conflict[i, j] is the conflict between i and j.
                # We want conf_counts[j] to equal the number of conflicts j has with S.
                # If we remove i, j has conf_counts[j] - Conflict[j, i] conflicts with S\{i}.
                # We want this to be 0.
                # So conf_counts[j] == Conflict[j, i].
                
                valid_js = np.where((~in_S) & (conf_counts == Conflict[i]))[0]
                
                if len(valid_js) < 2:
                    continue
                
                # We need to find a pair j1, j2 in valid_js such that Conflict[j1, j2] == 0
                # If valid_js is small, check all pairs.
                # If valid_js is large, we can pick a random subset or check first few.
                # Given the structure, valid_js is likely small.
                
                found_pair = False
                # Optimization: If valid_js is very large, just check a random sample
                if len(valid_js) > 20:
                    sample_idx = np.random.choice(len(valid_js), size=20, replace=False)
                    valid_js_check = valid_js[sample_idx]
                else:
                    valid_js_check = valid_js
                
                for p in range(len(valid_js_check)):
                    if found_pair:
                        break
                    j1 = valid_js_check[p]
                    # Check compatibility with other valid_js_check
                    # We want Conflict[j1, j2] == 0
                    # Get conflicts of j1 with valid_js_check
                    conflicts_with_j1 = Conflict[j1, valid_js_check]
                    # Find indices q where conflicts_with_j1[q] == 0 and q != p
                    compatible_qs = np.where(conflicts_with_j1 == 0)[0]
                    # Exclude p
                    compatible_qs = compatible_qs[compatible_qs != p]
                    if len(compatible_qs) > 0:
                        j2 = valid_js_check[compatible_qs[0]]
                        # Perform swap
                        S.remove(i)
                        S.add(j1)
                        S.add(j2)
                        
                        # Update in_S
                        in_S[i] = False
                        in_S[j1] = True
                        in_S[j2] = True
                        
                        # Update conf_counts
                        # conf_counts changes by: -Conflict[i] + Conflict[j1] + Conflict[j2]
                        conf_counts -= Conflict[i]
                        conf_counts += Conflict[j1]
                        conf_counts += Conflict[j2]
                        
                        # Reset counts for j1, j2 (they are now in S, so their count as candidates is irrelevant, 
                        # but for consistency, their count should be 0 if we consider them "not candidates"?
                        # Actually, conf_counts[j] is defined as sum of Conflict[j, k] for k in S.
                        # Since j1 is now in S, conf_counts[j1] includes Conflict[j1, j1]=0 and others.
                        # It doesn't matter as long as we mask out in_S when selecting candidates.
                        
                        improved = True
                        found_pair = True
                        break
                        
        return list(S)

    # Main loop: Try multiple seeds
    num_seeds = 200
    for s in range(num_seeds):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        res_improved = local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            # Optional: If we hit a high score, we can reduce seeds or increase local search depth
            # But let's just keep running within time limit.

    if not best_set:
        # Fallback
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
