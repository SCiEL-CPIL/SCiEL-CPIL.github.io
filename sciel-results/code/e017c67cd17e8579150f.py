"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit (5s buffer)
    
    # 1. Define the set of candidates
    candidates = list(range(2, 257))
    n = len(candidates)
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # 2. Precompute conflict masks
    # conflict_mask[i] is a bitmask of indices j such that lcm(candidates[i], candidates[j]) <= 256
    conflict_mask = [0] * n
    
    for i in range(n):
        a = candidates[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            b = candidates[j]
            if lcm(a, b) <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    # 3. Heuristic: Greedy + Local Search
    # Sort indices by weight (1/x) descending, which is x ascending
    # We want to pick small numbers first because they have larger weights.
    indices_by_weight = sorted(range(n), key=lambda i: candidates[i])
    
    def get_score(mask):
        # Calculate sum of 1/x for selected indices
        # This is slow if called often, so we maintain it incrementally in search
        pass
        
    def local_search(initial_mask):
        current_mask = initial_mask
        current_score = sum(1.0 / candidates[i] for i in range(n) if (current_mask >> i) & 1)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add any compatible vertex
            for i in indices_by_weight:
                if (current_mask >> i) & 1:
                    continue
                if (conflict_mask[i] & current_mask) == 0:
                    current_mask |= (1 << i)
                    current_score += 1.0 / candidates[i]
                    improved = True
                    break # Restart scan from best weight
            
            if not improved:
                # 2. Try 1-swap: remove one, add one
                # Iterate over current selected vertices
                selected_indices = [i for i in range(n) if (current_mask >> i) & 1]
                for i in selected_indices:
                    # Remove i
                    trial_mask = current_mask & ~(1 << i)
                    trial_score = current_score - 1.0 / candidates[i]
                    
                    # Find best j to add
                    best_gain = -1e18
                    best_j = -1
                    for j in indices_by_weight:
                        if (trial_mask >> j) & 1:
                            continue
                        if (conflict_mask[j] & trial_mask) == 0:
                            gain = 1.0 / candidates[j]
                            if gain > best_gain:
                                best_gain = gain
                                best_j = j
                                
                    if best_j != -1:
                        new_score = trial_score + best_gain
                        if new_score > current_score + 1e-12:
                            current_mask = trial_mask | (1 << best_j)
                            current_score = new_score
                            improved = True
                            break # Restart loop
                
                # 3. Try 1-2 swap: remove one, add two
                # This is O(N^3) per iteration, might be too slow if not careful.
                # We only do this if time permits.
                if time.time() < deadline - 20:
                    selected_indices = [i for i in range(n) if (current_mask >> i) & 1]
                    for i in selected_indices:
                        trial_mask = current_mask & ~(1 << i)
                        trial_score = current_score - 1.0 / candidates[i]
                        
                        # Find two best compatible vertices to add
                        # We need j1, j2 such that they are compatible with trial_mask and with each other.
                        # Iterate j1
                        best_pair_gain = -1e18
                        best_j1 = -1
                        best_j2 = -1
                        
                        # Optimization: Only consider j1 with high weight
                        for j1 in indices_by_weight:
                            if (trial_mask >> j1) & 1:
                                continue
                            if (conflict_mask[j1] & trial_mask) != 0:
                                continue
                            
                            # Now find j2 compatible with trial_mask | (1<<j1)
                            temp_mask = trial_mask | (1 << j1)
                            for j2 in indices_by_weight:
                                if j2 == j1:
                                    continue
                                if (temp_mask >> j2) & 1:
                                    continue
                                if (conflict_mask[j2] & temp_mask) != 0:
                                    continue
                                
                                gain = 1.0 / candidates[j1] + 1.0 / candidates[j2]
                                if gain > best_pair_gain:
                                    best_pair_gain = gain
                                    best_j1 = j1
                                    best_j2 = j2
                                    
                                    # Early break if we found a very good one? 
                                    # Since weights are 1/x, the best pair will involve small numbers.
                                    # We can break inner loop if 1/candidates[j2] is too small?
                                    # 1/j1 + 1/j2. If j1 is small, 1/j1 is large.
                                    # If we found a pair, further j2 will have smaller weight.
                                    # But j1 loop continues.
                                    break 
                            
                            # If best_j1 is found, can we break outer?
                            # No, a larger j1 might allow a much smaller j2?
                            # Unlikely, as smaller j1 usually allows more j2s.
                            # Let's just do a few iterations of outer loop.
                            if best_pair_gain > 0:
                                break
                        
                        if best_j1 != -1:
                            new_score = trial_score + best_pair_gain
                            if new_score > current_score + 1e-12:
                                current_mask = trial_mask | (1 << best_j1) | (1 << best_j2)
                                current_score = new_score
                                improved = True
                                break
                                
        return current_mask, current_score

    # Initial greedy mask
    greedy_mask = 0
    for i in indices_by_weight:
        if (conflict_mask[i] & greedy_mask) == 0:
            greedy_mask |= (1 << i)
            
    # Run local search on greedy result
    best_mask, best_score = local_search(greedy_mask)
    
    # 4. Try MILP if time allows
    # The MILP solver might be slow. We use the heuristic result as a benchmark.
    # If MILP finishes, it provides the exact optimum.
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        weights = np.array([1.0 / x for x in candidates])
        integrality = np.ones(n)
        c = -weights
        
        # Build constraints from conflict masks
        # We need x_i + x_j <= 1 for all conflicts.
        # To reduce constraints, we can use clique cover or just list all edges.
        # Number of edges is manageable.
        rows = []
        cols = []
        data = []
        
        for i in range(n):
            mask = conflict_mask[i]
            # Iterate set bits in mask
            # To avoid duplicates (i,j) and (j,i), only add if i < j
            j = 0
            m = mask
            while m:
                if m & 1:
                    if i < j:
                        rows.extend([len(data)//2, len(data)//2])
                        cols.extend([i, j])
                        data.extend([1, 1])
                m >>= 1
                j += 1
        
        if len(data) > 0:
            A = sp.coo_matrix((data, (rows, cols)), shape=(len(data)//2, n)).tocsr()
            b_ub = np.ones(len(data)//2)
            constraints = [LinearConstraint(A, ub=b_ub)]
        else:
            constraints = []
            
        bounds = Bounds(lb=0, ub=1)
        
        # Time limit check
        if time.time() < deadline - 10:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds
            )
            
            if res.status == 0:
                x_sol = res.x
                milp_mask = 0
                for i in range(n):
                    if x_sol[i] > 0.5:
                        milp_mask |= (1 << i)
                
                milp_score = sum(1.0 / candidates[i] for i in range(n) if (milp_mask >> i) & 1)
                
                if milp_score > best_score:
                    best_mask = milp_mask
                    best_score = milp_score
                else:
                    # MILP might have found a suboptimal solution if it timed out or hit precision issues?
                    # Or our heuristic was better? Unlikely for exact MILP.
                    # But we keep the best.
                    pass
        
    except Exception as e:
        # Fallback to heuristic result
        pass
        
    # 5. Final Polish on the best mask found
    final_mask, final_score = local_search(best_mask)
    
    # Extract result
    result = [candidates[i] for i in range(n) if (final_mask >> i) & 1]
    
    # Ensure non-empty
    if not result:
        # Fallback: pick 2
        result = [2]
        
    return result

# EVOLVE-BLOCK-END
