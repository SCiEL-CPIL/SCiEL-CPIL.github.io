import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Generate all APs within [0, n)
    aps = []
    for d in range(1, n):
        for a in range(d):
            seq = list(range(a, n, d))
            if len(seq) >= 2:
                aps.append(seq)
    
    m = len(aps)
    
    # Precompute for each index which APs it belongs to
    # Using lists of numpy arrays for fast access
    index_to_ap_ids = [np.array([], dtype=np.int64) for _ in range(n)]
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            if index_to_ap_ids[idx].size == 0:
                index_to_ap_ids[idx] = np.array([ap_id], dtype=np.int64)
            else:
                index_to_ap_ids[idx] = np.append(index_to_ap_ids[idx], ap_id)
    
    # Greedy Initialization
    signs = np.zeros(n, dtype=np.int8)
    ap_sums = np.zeros(m, dtype=np.int16)
    
    # To make greedy init faster, we can just use the same update logic
    # But we need to evaluate both +1 and -1 for each index
    # We will track the current max locally for the greedy step? 
    # No, the greedy step in inspirations minimized the *local* max of affected APs.
    # This is a good heuristic.
    
    for i in range(n):
        best_sign = 1
        best_local_max = float('inf')
        
        idxs = index_to_ap_ids[i]
        if idxs.size == 0:
            signs[i] = 1
            continue
            
        for s in [1, -1]:
            # Calculate new sums for affected APs
            # current sums for these APs are in ap_sums[idxs]
            # new sums = ap_sums[idxs] + s
            new_sums = ap_sums[idxs] + s
            local_max = np.max(np.abs(new_sums))
            
            if local_max < best_local_max:
                best_local_max = local_max
                best_sign = s
        
        signs[i] = best_sign
        ap_sums[idxs] += best_sign
    
    current_max = np.max(np.abs(ap_sums))
    best_signs = signs.copy()
    best_max = current_max
    
    # Local Search: Hill Climbing with Critical AP Optimization
    # We only consider flips that affect at least one AP with |sum| == current_max
    # because only those can potentially reduce the global maximum.
    
    no_improve_count = 0
    max_no_improve = 100 # Allow more iterations since checks are cheaper on average
    
    while time.time() < deadline - 5:
        # Identify critical APs
        critical_mask = np.abs(ap_sums) == current_max
        critical_aps = np.nonzero(critical_mask)[0]
        
        if len(critical_aps) == 0:
            break # Should not happen if current_max > 0
            
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 2:
                break
            
            idxs = index_to_ap_ids[i]
            if idxs.size == 0:
                continue
            
            # Check if this flip affects any critical AP
            # Intersection of idxs and critical_aps
            # Using np.intersect1d is slow. 
            # Better: check if any of idxs is in critical_aps.
            # Since critical_aps is a list of indices into ap_sums, and idxs is also indices into ap_sums.
            # We can create a boolean mask for critical_aps? 
            # Or simply: if not np.any(np.isin(idxs, critical_aps)): continue
            # np.isin is O(N*M). 
            # Alternative: Maintain a set of critical AP ids?
            # Let's use a boolean array for critical status, updated when max changes.
            
            # For now, let's just check intersection. 
            # If m is ~30k, and idxs is ~256, this is fast.
            if not np.any(np.isin(idxs, critical_aps)):
                continue
                
            # Try flip
            old_sign = signs[i]
            new_sign = -old_sign
            signs[i] = new_sign
            
            old_sums = ap_sums[idxs].copy()
            ap_sums[idxs] += 2 * new_sign
            
            # Check if global max decreased
            # New local max
            local_max = np.max(np.abs(ap_sums[idxs]))
            
            # The new global max is max(local_max, max of unaffected APs)
            # We don't know max of unaffected APs without checking.
            # But if local_max < current_max, it's a candidate for improvement.
            # We must verify if the *global* max actually dropped.
            # The global max could still be current_max if an unaffected AP has that sum.
            # So we need to compute the true new global max.
            new_global_max = np.max(np.abs(ap_sums))
            
            if new_global_max < current_max:
                # Accept
                current_max = new_global_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                ap_sums[idxs] = old_sums
                signs[i] = old_sign
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart with perturbation
                if best_max > 2: # If we can't get below 2, try harder
                    signs = best_signs.copy()
                    # Perturb 5-10 random positions
                    num_perturb = random.randint(5, 10)
                    for _ in range(num_perturb):
                        idx = random.randint(0, n-1)
                        signs[idx] = -signs[idx]
                    # Recompute sums
                    ap_sums = np.zeros(m, dtype=np.int16)
                    for ap_id, ap in enumerate(aps):
                        s = 0
                        for idx in ap:
                            s += signs[idx]
                        ap_sums[ap_id] = s
                    current_max = np.max(np.abs(ap_sums))
                    no_improve_count = 0
                else:
                    break

    return best_signs.tolist()
