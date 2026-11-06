import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # 1. Precompute APs and incidence structure
    # We need:
    # - List of all APs (to compute initial sums and hard max)
    # - For each index i, list of AP indices that contain i (for fast updates)
    
    ap_indices = []
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                ap_indices.append(terms)
                
    num_aps = len(ap_indices)
    
    # Build reverse index: aps_containing[i] = list of ap indices j where i in ap_indices[j]
    aps_containing = [[] for _ in range(N)]
    for j, terms in enumerate(ap_indices):
        for i in terms:
            aps_containing[i].append(j)
            
    # Helper to compute initial sums for a given sign vector
    def compute_initial_sums(signs):
        S = np.zeros(num_aps, dtype=np.float64)
        for j, terms in enumerate(ap_indices):
            # Sum signs[terms]
            # Using np.sum on a list of indices is slower than slicing, but terms are sparse.
            # Actually, for d=1, terms are contiguous. For d>1, they are strided.
            # A loop in Python might be slow. Let's use numpy.
            # S[j] = np.sum(signs[terms])
            # This creates a temporary array. 
            # Alternative:
            # S[j] = 0
            # for i in terms: S[j] += signs[i]
            # Since num_aps is ~32k and avg len is ~2, this is fine?
            # No, avg len is small for large d, large for small d.
            # d=1: len up to 256.
            # Let's just do it efficiently.
            S[j] = np.sum(signs[terms])
        return S

    # Helper to compute soft objective
    def calc_soft_obj(S, beta):
        abs_S = np.abs(S)
        max_val = np.max(abs_S)
        if max_val == 0:
            return 0.0
        # Shift to avoid overflow
        shifted = abs_S / beta - max_val / beta
        sum_exp = np.sum(np.exp(shifted))
        return beta * (max_val / beta + np.log(sum_exp))
        
    # Helper to get hard max
    def get_hard_max(S):
        return int(np.max(np.abs(S)))

    best_signs = None
    best_hard_max = N # Worst case

    # Multi-restart SA
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        
        # Initialize signs
        # Try Thue-Morse for first few, then random
        if seed < 3:
            # Thue-Morse
            signs = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
        else:
            signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
            
        S = compute_initial_sums(signs)
        current_hard_max = get_hard_max(S)
        
        if current_hard_max < best_hard_max:
            best_hard_max = current_hard_max
            best_signs = signs.copy()
            
        # SA Parameters
        beta = 1.0
        T = 5.0
        
        current_obj = calc_soft_obj(S, beta)
        
        # SA Loop
        steps = 0
        max_steps = 200000 # High number, limited by time
        
        while time.time() < deadline - 2:
            steps += 1
            
            # Pick random index to flip
            i = rng.randint(0, N - 1)
            old_sign = signs[i]
            
            # Update sums for APs containing i
            # Delta S_j = -2 * old_sign
            delta_s = -2.0 * old_sign
            
            # We need to update S in place for the objective calculation, 
            # but we also need to revert if rejected.
            # So we store the old values or compute the new objective from scratch?
            # Computing from scratch is O(num_aps). 
            # Updating delta is O(len(aps_containing[i])).
            # len(aps_containing[i]) is ~128.
            # num_aps is ~32000.
            # So incremental update is much faster for the state S.
            # But we need to calculate the NEW soft objective.
            # Soft obj depends on ALL S.
            # If we update S in place, we can calculate new_obj.
            # If rejected, we revert S.
            
            # Store old S values for affected APs?
            # Or just copy the whole S array? 32k floats is ~256KB. Copying is fast.
            # Let's try in-place update with revert.
            
            affected_aps = aps_containing[i]
            # Apply change
            S[affected_aps] += delta_s
            
            new_obj = calc_soft_obj(S, beta)
            delta_obj = new_obj - current_obj
            
            accept = False
            if delta_obj < 0:
                accept = True
            else:
                if delta_obj / T > 100:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_obj / T)
                    
            if accept:
                signs[i] = -old_sign
                current_obj = new_obj
                # Update hard max check periodically
                if steps % 100 == 0:
                    current_hard_max = get_hard_max(S)
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
            else:
                # Revert S
                S[affected_aps] -= delta_s
                
            # Annealing
            if steps % 1000 == 0:
                T *= 0.95
                beta *= 0.98
                if beta < 0.1:
                    beta = 0.1
                if T < 0.1:
                    T = 0.1
                    
            # Early stop if optimal
            if best_hard_max <= 1:
                break
                
    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
