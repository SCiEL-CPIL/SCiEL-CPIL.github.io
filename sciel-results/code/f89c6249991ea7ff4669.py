import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    # Precompute APs
    # AP is defined by start s and step d.
    # Indices: s, s+d, ..., < n
    # We store APs as lists of indices.
    # We also need a mapping from index to list of AP indices containing it.
    aps = []
    index_to_aps = [[] for _ in range(n)]
    
    # Optimization: Only store APs of length >= 1.
    # The problem states "including singletons".
    # Singleton APs have sum +/-1, so they don't constrain the max if max >= 1.
    # But we must include them for correctness of the "any AP" definition if we were checking exact matches.
    # For minimization of max, singletons are trivial.
    # We will include all APs to be safe, but the max will be driven by longer ones.
    
    for d in range(1, n):
        for s in range(n):
            idxs = list(range(s, n, d))
            if idxs:
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)
    
    num_aps = len(aps)
    
    # Helper to calculate max absolute sum from ap_sums
    def get_max_abs(ap_sums):
        return int(np.max(np.abs(ap_sums)))

    # --- Phase 1: Greedy Min-Max Construction ---
    # This builds a solution that locally minimizes the peak discrepancy at each step.
    
    signs = np.zeros(n, dtype=np.int8)
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    
    # We need to efficiently evaluate the impact of setting signs[i].
    # When we set signs[i], we affect all APs in index_to_aps[i].
    # We want to choose signs[i] such that the new max abs sum is minimized.
    
    # To speed up the greedy phase, we can just compute the sums for the affected APs.
    # Since n=256, the number of APs is roughly n^2/2 ~ 32k.
    # For each i, we check 2 options.
    # Cost: n * 2 * (avg APs per index).
    # Avg APs per index is small.
    
    for i in range(n):
        affected_aps = index_to_aps[i]
        
        best_sign = 1
        best_max_val = float('inf')
        
        # Try +1
        # We need to simulate the change.
        # Current ap_sums does not include signs[i] yet (it's 0).
        # So new_sum = old_sum + sign.
        
        max_plus = 0
        max_minus = 0
        
        # We need to check the resulting max for the *entire* set of APs?
        # No, only the affected APs change. The unaffected ones stay the same.
        # So we need to know the current global max of unaffected APs.
        # Let current_global_max_unaffected = max(abs(ap_sums[j]) for j not in affected_aps).
        # This is expensive to compute from scratch each time.
        
        # Alternative: Just compute the max for affected APs and compare with a tracked global max?
        # Actually, the greedy choice depends on the global max.
        # If the global max is 5, and changing i makes an affected AP 4, the global max stays 5.
        # If changing i makes an affected AP 6, the global max becomes 6.
        # So we want to minimize the new global max.
        
        # Let M_unaffected be the max abs sum of APs NOT containing i.
        # Let M_affected_plus be the max abs sum of APs containing i, assuming sign=+1.
        # Let M_affected_minus be the max abs sum of APs containing i, assuming sign=-1.
        
        # New Global Max (Plus) = max(M_unaffected, M_affected_plus)
        # New Global Max (Minus) = max(M_unaffected, M_affected_minus)
        
        # We can compute M_unaffected once per step i?
        # No, M_unaffected changes as we fill in previous signs.
        # But for a fixed i, M_unaffected is constant for the choice between +1 and -1.
        # So we just need to compare M_affected_plus and M_affected_minus.
        # Wait, if both are less than M_unaffected, the global max is M_unaffected regardless.
        # In that case, we can pick randomly or based on future potential.
        # If one is greater than M_unaffected, we prefer the one that is smaller (or both are > M_unaffected, pick the smaller).
        
        # So the decision rule is:
        # Calculate max_abs_plus = max(abs(ap_sums[k] + 1) for k in affected_aps)
        # Calculate max_abs_minus = max(abs(ap_sums[k] - 1) for k in affected_aps)
        
        # If max_abs_plus < max_abs_minus: choose +1
        # If max_abs_minus < max_abs_plus: choose -1
        # If equal: choose randomly (or look ahead? keep it simple: random)
        
        # Note: ap_sums currently holds the sum of already assigned signs.
        # signs[i] is currently 0.
        
        if affected_aps:
            # Vectorized calculation
            old_vals = ap_sums[affected_aps]
            
            # New values if we add +1
            new_vals_plus = old_vals + 1
            max_plus = int(np.max(np.abs(new_vals_plus)))
            
            # New values if we add -1
            new_vals_minus = old_vals - 1
            max_minus = int(np.max(np.abs(new_vals_minus)))
            
            if max_plus < max_minus:
                best_sign = 1
            elif max_minus < max_plus:
                best_sign = -1
            else:
                # Tie-breaking: try to balance the distribution?
                # Just pick randomly for now to avoid bias
                best_sign = 1 if random.random() < 0.5 else -1
        else:
            best_sign = 1 if random.random() < 0.5 else -1
            
        signs[i] = best_sign
        # Update ap_sums
        ap_sums[affected_aps] += best_sign

    # Initial best from Greedy
    current_max = get_max_abs(ap_sums)
    best_signs = signs.copy()
    best_max = current_max
    
    # --- Phase 2: Simulated Annealing ---
    # Refine the greedy solution.
    
    rng = random.Random(42)
    
    # Frequency of absolute sums for O(1) max tracking?
    # Max possible sum is 256.
    max_possible = n
    freq_abs = np.zeros(max_possible + 1, dtype=np.int32)
    
    for s in ap_sums:
        freq_abs[abs(s)] += 1
        
    T = 10.0
    cooling_rate = 0.9995
    steps = 0
    
    # We will use a mix of single-bit and double-bit flips
    # Double-bit flips help escape local minima where single flips are neutral or bad.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Choose move type
        # 80% single, 20% double
        if rng.random() < 0.8:
            # Single bit flip
            i = rng.randrange(n)
            old_sign = signs[i]
            delta = -2 * old_sign
            
            affected_aps = index_to_aps[i]
            
            if not affected_aps:
                continue
                
            old_vals = ap_sums[affected_aps]
            new_vals = old_vals + delta
            
            # Calculate new max
            new_max_candidate = current_max
            
            # Update freq and find new max
            # We need to be careful with the max tracking.
            # Let's just compute the new max for affected and compare with current global max.
            # Actually, updating freq is faster if we do it right.
            
            # Let's stick to the freq array update for robustness.
            # Decrement old, increment new.
            
            local_max_increase = False
            for idx_ap, old_v, new_v in zip(affected_aps, old_vals, new_vals):
                oa = abs(old_v)
                na = abs(new_v)
                
                freq_abs[oa] -= 1
                freq_abs[na] += 1
                
                ap_sums[idx_ap] = new_v
                
                if na > new_max_candidate:
                    new_max_candidate = na
            
            # Determine actual new max
            actual_new_max = new_max_candidate
            if freq_abs[current_max] == 0:
                # Max dropped
                while actual_new_max > 0 and freq_abs[actual_new_max] == 0:
                    actual_new_max -= 1
            
            cost_diff = actual_new_max - current_max
            
            if cost_diff < 0:
                accept = True
            else:
                if T < 1e-5:
                    accept = False
                else:
                    if cost_diff / T > 50:
                        prob = 0.0
                    else:
                        prob = math.exp(-cost_diff / T)
                    accept = rng.random() < prob
            
            if accept:
                signs[i] = -signs[i]
                current_max = actual_new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                ap_sums[affected_aps] = old_vals
                for idx_ap, old_v, new_v in zip(affected_aps, old_vals, new_vals):
                    oa = abs(old_v)
                    na = abs(new_v)
                    freq_abs[na] -= 1
                    freq_abs[oa] += 1
            
        else:
            # Double bit flip
            # Pick two distinct indices
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            while i2 == i1:
                i2 = rng.randrange(n)
                
            s1 = signs[i1]
            s2 = signs[i2]
            
            # Deltas
            d1 = -2 * s1
            d2 = -2 * s2
            
            aff1 = index_to_aps[i1]
            aff2 = index_to_aps[i2]
            
            # Union of affected APs
            # Since n is small, we can just iterate.
            # We need to update sums for APs in aff1 (delta d1) and aff2 (delta d2).
            # If an AP is in both, it gets d1 + d2.
            
            # Map AP index to total delta
            delta_map = {}
            for ap in aff1:
                delta_map[ap] = delta_map.get(ap, 0) + d1
            for ap in aff2:
                delta_map[ap] = delta_map.get(ap, 0) + d2
                
            if not delta_map:
                continue
                
            affected_aps_list = list(delta_map.keys())
            
            old_vals = ap_sums[affected_aps_list]
            
            # Calculate new vals
            new_vals = np.empty_like(old_vals)
            for idx, ap_id in enumerate(affected_aps_list):
                new_vals[idx] = old_vals[idx] + delta_map[ap_id]
                
            new_max_candidate = current_max
            
            # Update freq
            for idx_ap, old_v, new_v in zip(affected_aps_list, old_vals, new_vals):
                oa = abs(old_v)
                na = abs(new_v)
                
                freq_abs[oa] -= 1
                freq_abs[na] += 1
                
                ap_sums[idx_ap] = new_v
                
                if na > new_max_candidate:
                    new_max_candidate = na
            
            actual_new_max = new_max_candidate
            if freq_abs[current_max] == 0:
                while actual_new_max > 0 and freq_abs[actual_new_max] == 0:
                    actual_new_max -= 1
            
            cost_diff = actual_new_max - current_max
            
            if cost_diff < 0:
                accept = True
            else:
                if T < 1e-5:
                    accept = False
                else:
                    if cost_diff / T > 50:
                        prob = 0.0
                    else:
                        prob = math.exp(-cost_diff / T)
                    accept = rng.random() < prob
            
            if accept:
                signs[i1] = -signs[i1]
                signs[i2] = -signs[i2]
                current_max = actual_new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                ap_sums[affected_aps_list] = old_vals
                for idx_ap, old_v, new_v in zip(affected_aps_list, old_vals, new_vals):
                    oa = abs(old_v)
                    na = abs(new_v)
                    freq_abs[na] -= 1
                    freq_abs[oa] += 1

        T *= cooling_rate
        steps += 1
        
        # Optional: Restart if stuck?
        # For now, rely on the mix of moves.
        
    return best_signs.tolist()
