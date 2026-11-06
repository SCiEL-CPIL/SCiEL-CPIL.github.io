import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Generate all arithmetic progressions (length >= 2)
    # The task says "including one", but singletons have sum +/-1 always, 
    # so they don't affect the maximization if we find a solution with max > 1.
    # If max is 1, singletons are fine. If max > 1, singletons are irrelevant.
    # We focus on length >= 2.
    ap_indices = []
    for start in range(n):  # 0-indexed start
        for step in range(1, n):
            terms = []
            idx = start
            while idx < n:
                terms.append(idx)
                idx += step
            if len(terms) >= 2:
                ap_indices.append(np.array(terms, dtype=np.int32))
                
    num_aps = len(ap_indices)
    
    # 2. Build sparse representation for fast sum updates
    # For each position i, store which APs it belongs to.
    # We need to update ap_sums efficiently when flipping a sign.
    # Flipping sign at pos i changes sum of AP k by 2 * sign[i] (old) -> -2 * sign[i] (new)?
    # Let s_i be the sign. Sum_k = sum_{j in AP_k} s_j.
    # If we flip s_i to -s_i, new_sum_k = sum_k - 2*s_i.
    
    # pos_to_aps[i] = list of AP indices containing i
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idxs in enumerate(ap_indices):
        for i in idxs:
            pos_to_aps[i].append(ap_id)
            
    # Convert to numpy arrays for faster iteration if needed, 
    # but lists are often faster for small counts in Python loops.
    # However, we want to vectorize the *evaluation* of flips.
    
    # 3. Initialize signs
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=n).astype(np.int8)
    
    # Compute initial AP sums
    # ap_sums[k] = sum(signs[idx] for idx in ap_indices[k])
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for k in range(num_aps):
        ap_sums[k] = np.sum(signs[ap_indices[k]])
        
    current_max = np.max(np.abs(ap_sums))
    best_signs = signs.copy()
    best_max = current_max
    
    # 4. Simulated Annealing
    # Objective: Minimize max(|ap_sums|)
    # We use a soft penalty: cost = max_abs_sum + lambda * (sum of squares)
    # Or just track max and accept if max decreases, or with prob exp(-delta/T) if max increases.
    # To make it vectorized:
    # For each position i, calculate the change in max if we flip i.
    # This is hard to do in one vectorized pass because max is non-linear.
    # However, we can compute the new max for *each* candidate flip i.
    # new_sums_i = ap_sums.copy()
    # new_sums_i[pos_to_aps[i]] -= 2 * signs[i]
    # new_max_i = max(abs(new_sums_i))
    # This is O(n * num_aps) per step if done naively, which is 256 * ~2000 = 500k ops.
    # 500k ops in NumPy is fast, but doing it 256 times in a loop is slow.
    
    # Alternative: 
    # We can precompute the impact of flipping each position on the AP sums.
    # impact[i] is a sparse vector of length num_aps.
    # But we need the MAX of the absolute sums.
    
    # Let's try a different vectorization:
    # We maintain ap_sums.
    # We want to find i that minimizes new_max.
    # new_max_i = max( max(abs(ap_sums[not_affected_i])), max(abs(ap_sums[affected_i] - 2*signs[i])) )
    # This is still complex to vectorize fully across all i simultaneously without huge memory.
    
    # Let's stick to a semi-vectorized approach or a fast Python loop if NumPy overhead is too high.
    # Actually, 256 flips * 2000 APs = 512k operations. 
    # If we use NumPy arrays for ap_sums, updating a subset is fast.
    # But calculating max(abs(...)) on the whole array for each i is O(num_aps).
    # So 256 * 2000 = 512,000 operations per step.
    # 50,000 steps * 512,000 ops = 2.5e10 ops. Too slow.
    
    # We need a faster evaluation.
    # Observation: Most flips do NOT change the global max.
    # Only flips that affect the APs currently at the max (or near it) matter.
    # Let M = current_max.
    # Let critical_aps = indices where abs(ap_sums) == M.
    # If we flip i, and i is NOT in any critical AP, the max remains M (unless it creates a new larger max, which is unlikely if we start low, but possible).
    # Actually, if we flip i, only APs containing i change.
    # If none of those APs are critical, the new max is max(M, max_new_affected).
    # If M is high, we only care about reducing it.
    
    # Strategy:
    # 1. Find current max M.
    # 2. Identify critical APs (abs(sum) == M).
    # 3. Identify positions i that belong to at least one critical AP.
    # 4. For each such i, calculate new max.
    # 5. If no critical APs (M=0 or 1?), we are in a basin. Use random moves.
    
    # Let's refine the "critical" logic.
    # If we flip i, the new sums for affected APs are s_k - 2*signs[i].
    # The new global max is max( max_abs_unaffected, max_abs_affected_new ).
    # max_abs_unaffected is M unless all APs with abs(sum)=M are affected by i.
    # If all critical APs are affected, M might drop.
    
    # Let's implement a fast local search first, then SA.
    
    # Precompute pos_to_aps as lists of numpy arrays? No, lists of ints.
    # Let's try to optimize the inner loop.
    
    best_signs = signs.copy()
    best_max = current_max
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    # We will do a number of steps based on time.
    # Each step:
    # 1. Find max and critical APs.
    # 2. Candidate positions: those in critical APs. If none, random.
    # 3. Evaluate candidates.
    # 4. Pick best or random with SA probability.
    
    while time.time() < deadline - 2.0:
        # Find current max
        abs_sums = np.abs(ap_sums)
        current_max = np.max(abs_sums)
        
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
            
        # If we reach 1, we can stop? No, 0 is impossible for n>1? 
        # Actually, can we get 0? No, sum of odd length AP is odd, even is even.
        # Min max is likely 1 or 2.
        
        # Find critical APs
        critical_mask = (abs_sums == current_max)
        critical_ap_ids = np.where(critical_mask)[0]
        
        if len(critical_ap_ids) == 0:
            # Should not happen if current_max is max
            break
            
        # Find positions involved in critical APs
        candidate_positions = set()
        for ap_id in critical_ap_ids:
            for pos in ap_indices[ap_id]:
                candidate_positions.add(pos)
                
        # Also, we might want to try random positions to escape local minima
        # But let's stick to critical ones first for efficiency.
        # If we are stuck (no improvement in last X steps), add random.
        
        if len(candidate_positions) == 0:
            candidate_positions = {rng.integers(0, n)}
            
        # Evaluate each candidate position
        # We want to minimize new_max.
        best_candidate_pos = -1
        best_candidate_max = current_max + 1 # infinity
        
        # Limit candidates to avoid too much work if there are many
        # But critical APs are usually few.
        # Let's convert to list
        cand_list = list(candidate_positions)
        
        # Shuffle to break ties
        rng.shuffle(cand_list)
        
        for i in cand_list:
            # Calculate new max if we flip i
            # Affected APs
            affected_aps = pos_to_aps[i]
            if not affected_aps:
                new_max = current_max
            else:
                # Get current sums of affected APs
                aff_sums = ap_sums[affected_aps]
                # New sums
                new_aff_sums = aff_sums - 2 * signs[i]
                # Max of affected new sums
                max_aff_new = np.max(np.abs(new_aff_sums))
                
                # Max of unaffected sums
                # If there are other critical APs not in affected_aps, max is still current_max
                # If all critical APs are in affected_aps, we need to check if any other AP has sum == current_max
                # This check is expensive. 
                # Approximation: Assume max_unaffected = current_max unless we are very sure.
                # Actually, if we flip i, and i is in ALL critical APs, then the old critical value is gone.
                # The new max is max(max_aff_new, max_other).
                # max_other <= current_max.
                # If max_aff_new < current_max, new_max might be < current_max.
                # If max_aff_new >= current_max, new_max >= current_max.
                
                # Let's just compute the full max for simplicity if num_aps is small enough?
                # num_aps is ~2000. 
                # Doing this for ~10 candidates is 20k ops. Fast.
                # Let's do it properly.
                
                # Copy ap_sums? No, too slow.
                # We can calculate max of all APs.
                # new_sums = ap_sums.copy()
                # new_sums[affected_aps] = new_aff_sums
                # new_max = np.max(np.abs(new_sums))
                
                # This is O(num_aps). 
                # 10 candidates * 2000 = 20k ops.
                # 50k steps * 20k = 1e9 ops. Might be tight in 50s.
                # Let's optimize:
                # We know max_other <= current_max.
                # new_max = max(max_aff_new, max_other)
                # If max_aff_new < current_max:
                #    We need to know if any other AP has abs(sum) == current_max.
                #    If yes, new_max = current_max.
                #    If no, new_max = max(max_aff_new, second_max_other).
                #    This is complex.
                
                # Let's just use the copy method but only for the top few candidates?
                # Or just do it for all. 20k ops is ~1-2 ms in NumPy? 
                # np.max on 2000 ints is very fast.
                # Let's try it.
                
                # Optimization: 
                # If max_aff_new >= current_max:
                #    new_max = max_aff_new (since it's >= old max, and others <= old max)
                #    Wait, others could be > max_aff_new? No, others <= current_max <= max_aff_new.
                #    So if max_aff_new >= current_max, new_max = max_aff_new.
                # If max_aff_new < current_max:
                #    new_max is either current_max (if some other AP has current_max) or max(max_aff_new, max_other).
                #    We can check if any non-affected AP has abs(sum) == current_max.
                
                if max_aff_new >= current_max:
                    new_max = max_aff_new
                else:
                    # Check if any non-affected critical AP exists
                    # critical_mask is bool array of length num_aps
                    # affected_mask is False everywhere, True at affected_aps
                    # We need to check if critical_mask & ~affected_mask has any True
                    # This is O(num_aps) to build affected_mask? 
                    # We can precompute a boolean array for affected?
                    # Or just check: 
                    # Are there any critical APs NOT in affected_aps?
                    # critical_ap_ids is small. affected_aps is small.
                    # intersection = set(critical_ap_ids) & set(affected_aps)
                    # if len(intersection) < len(critical_ap_ids):
                    #    new_max = current_max
                    # else:
                    #    # All critical APs are affected.
                    #    # Need to find max of all other APs.
                    #    # This requires scanning all APs.
                    #    # But this case is rare (flipping a position that hits ALL critical APs).
                    #    # Let's just do the full scan in this rare case.
                    #    pass
                    
                    # Let's use the set intersection check
                    aff_set = set(affected_aps)
                    if not set(critical_ap_ids).issubset(aff_set):
                        new_max = current_max
                    else:
                        # Rare case: full scan
                        # Create a view? 
                        # new_sums = ap_sums.copy()
                        # new_sums[affected_aps] = new_aff_sums
                        # new_max = np.max(np.abs(new_sums))
                        # This is acceptable if rare.
                        new_sums = ap_sums.copy()
                        new_sums[affected_aps] = new_aff_sums
                        new_max = np.max(np.abs(new_sums))
                        
            if new_max < best_candidate_max:
                best_candidate_max = new_max
                best_candidate_pos = i
                
        # Select move
        if best_candidate_pos != -1:
            # Accept best candidate? Or SA?
            # Let's use SA to escape local minima.
            # If best_candidate_max < current_max: Accept
            # If best_candidate_max > current_max: Accept with prob exp(-delta/T)
            # If equal: Accept with prob 0.5?
            
            delta = best_candidate_max - current_max
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < np.exp(-delta / T)
                
            if accept:
                # Apply flip
                i = best_candidate_pos
                signs[i] = -signs[i]
                # Update ap_sums
                for ap_id in pos_to_aps[i]:
                    ap_sums[ap_id] -= 2 * (-signs[i]) # Wait, signs[i] is already flipped?
                    # No, signs[i] is updated. 
                    # Old sign was s. New sign is -s.
                    # Change in sum is -2*s_old = 2*s_new.
                    # ap_sums[ap_id] += 2 * signs[i] (new sign)
                    # Let's re-verify.
                    # sum_new = sum_old - s_old + s_new = sum_old - s_old - s_old = sum_old - 2*s_old.
                    # s_old = -signs[i] (current signs[i] is new).
                    # sum_new = sum_old - 2*(-signs[i]) = sum_old + 2*signs[i].
                    # So ap_sums[ap_id] += 2 * signs[i] is correct.
                    pass # The loop below does it
                
                # Actually, I didn't update ap_sums in the code above. I just calculated new_max.
                # I need to update ap_sums if I accept.
                for ap_id in pos_to_aps[i]:
                    ap_sums[ap_id] += 2 * signs[i]
                    
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

    return [int(s) for s in best_signs]
