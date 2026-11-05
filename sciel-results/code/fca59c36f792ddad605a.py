"""Low discrepancy on all arithmetic progressions - Targeted Local Search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_ap_structure(n):
    """
    Build list of all arithmetic progressions with length >= 2.
    Returns:
        aps: List of lists of indices. aps[i] is the list of indices for the i-th AP.
        ap_membership: List of lists. ap_membership[i] is the list of AP indices that contain element i.
    """
    aps = []
    ap_membership = [[] for _ in range(n)]
    
    # Step d from 1 to n-1
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)
                
    return aps, ap_membership

def _max_ap_discrepancy(signs, aps):
    """
    Compute max absolute signed sum over any arithmetic progression.
    signs: numpy array of +1/-1.
    aps: list of AP index lists.
    Returns the max absolute sum.
    """
    max_abs = 1 # Singletons always have abs sum 1
    
    for ap in aps:
        # Extract values
        vals = signs[ap]
        # Cumulative sum
        cum = np.concatenate(([0], np.cumsum(vals)))
        
        # Max subarray sum
        # We want max(cum[j] - cum[i]) for j > i
        # This is equivalent to max(cum) - min(cum[:j])
        # Using prefix min/max
        min_prefix = np.minimum.accumulate(cum)
        max_prefix = np.maximum.accumulate(cum)
        
        # Max sum ending at j is cum[j] - min(cum[0..j-1])
        # But we can just use: max(cum - min_prefix) is not quite right for all j simultaneously in one shot without care.
        # Actually, max subarray sum is max(cum[j] - min(cum[0..j-1])).
        # Let's do it simply for small APs (most are small).
        # For speed, we can just iterate if AP is small, or use vectorized ops if large.
        # Most APs are short. Let's stick to a robust vectorized approach.
        
        # min_prefix[k] is min of cum[0..k]
        # We need min of cum[0..j-1] for each j. 
        # So we shift min_prefix by 1? 
        # min_prefix_before = np.concatenate(([0], min_prefix[:-1]))
        # max_sub = np.max(cum - min_prefix_before)
        
        min_prefix_before = np.concatenate(([0], min_prefix[:-1]))
        max_sub = np.max(cum - min_prefix_before)
        
        max_prefix_before = np.concatenate(([0], max_prefix[:-1]))
        min_sub = np.min(cum - max_prefix_before)
        
        current_max = max(abs(int(max_sub)), abs(int(min_sub)))
        if current_max > max_abs:
            max_abs = current_max
            
    return max_abs

def _find_worst_aps(signs, aps, current_disc):
    """
    Find APs that contribute to the current maximum discrepancy.
    Returns a list of AP indices.
    """
    worst_aps = []
    for i, ap in enumerate(aps):
        vals = signs[ap]
        cum = np.concatenate(([0], np.cumsum(vals)))
        min_prefix = np.minimum.accumulate(cum)
        max_prefix = np.maximum.accumulate(cum)
        
        min_prefix_before = np.concatenate(([0], min_prefix[:-1]))
        max_sub = np.max(cum - min_prefix_before)
        
        max_prefix_before = np.concatenate(([0], max_prefix[:-1]))
        min_sub = np.min(cum - max_prefix_before)
        
        if abs(int(max_sub)) == current_disc or abs(int(min_sub)) == current_disc:
            worst_aps.append(i)
            
    return worst_aps

def construct():
    deadline = time.time() + 55  # 55s safety margin for 60s limit (or 90s if that's the hard kill, but 55 is safe)
    n = 256
    rng = random.Random(42)
    
    # Build AP structure
    aps, ap_membership = _build_ap_structure(n)
    
    # Initial Candidates
    candidates = []
    
    # 1. Thue-Morse
    signs_tm = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    candidates.append(signs_tm)
    
    # 2. Alternating
    signs_alt = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    candidates.append(signs_alt)
    
    # 3. Random Seeds
    for seed in range(10):
        rng_c = random.Random(seed)
        candidates.append(np.array([rng_c.choice((-1, 1)) for _ in range(n)], dtype=np.int8))
        
    # Evaluate best initial
    best_signs = candidates[0]
    best_val = _max_ap_discrepancy(best_signs, aps)
    
    for c in candidates[1:]:
        if time.time() > deadline - 20:
            break
        val = _max_ap_discrepancy(c, aps)
        if val < best_val:
            best_val = val
            best_signs = c
    
    current_signs = best_signs.copy()
    current_val = best_val
    
    # Local Search Loop
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() < deadline - 5:
        if no_improve_count >= max_no_improve:
            # Random restart or shake
            if rng.random() < 0.5:
                # Full restart
                current_signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)
                current_val = _max_ap_discrepancy(current_signs, aps)
            else:
                # Shake: flip 5-10 random signs
                for _ in range(rng.randint(5, 10)):
                    idx = rng.randint(0, n-1)
                    current_signs[idx] *= -1
                current_val = _max_ap_discrepancy(current_signs, aps)
            
            if current_val < best_val:
                best_val = current_val
                best_signs = current_signs.copy()
            
            no_improve_count = 0
            continue
        
        # Targeted Descent
        # Find worst APs
        worst_aps = _find_worst_aps(current_signs, aps, current_val)
        
        # If no worst APs found (shouldn't happen if val > 1), break or random
        if not worst_aps:
            no_improve_count += 1
            continue
            
        # Collect candidate flips from worst APs
        # To avoid duplicates, use a set of indices to try
        candidate_indices = set()
        for ap_idx in worst_aps:
            ap = aps[ap_idx]
            # Sample up to 5 indices from this AP to try flipping
            # If AP is small, try all
            if len(ap) <= 5:
                candidate_indices.update(ap)
            else:
                # Try first, last, and middle
                candidate_indices.add(ap[0])
                candidate_indices.add(ap[-1])
                candidate_indices.add(ap[len(ap)//2])
                # Add 2 random
                random_indices = rng.sample(ap, 2)
                candidate_indices.update(random_indices)
                
        improved = False
        best_flip_idx = -1
        best_flip_val = current_val
        
        # Try flipping each candidate
        for idx in candidate_indices:
            # Flip
            current_signs[idx] *= -1
            # Evaluate
            # Optimization: We only need to check if the new max is < current_val
            # But _max_ap_discrepancy is O(N_APS * AvgLen). 
            # We can do a quick check on the affected APs?
            # For simplicity and robustness, just recompute full if it looks promising?
            # Actually, recomputing full is ~50-100ms. 
            # If we have ~10-20 candidates, that's ~1-2s per iteration. 
            # This might be too slow for many iterations.
            
            # Let's do a faster check:
            # Only recompute discrepancy for APs that contain idx?
            # And check if any OTHER AP exceeds current_val?
            # This is complex. Let's just recompute full but limit candidates.
            
            val = _max_ap_discrepancy(current_signs, aps)
            
            if val < best_flip_val:
                best_flip_val = val
                best_flip_idx = idx
            else:
                # Revert immediately if not better
                current_signs[idx] *= -1
                continue
            # If better, we keep the flip for now to check if an even better one exists?
            # No, steepest descent: find the BEST single flip.
            # So we must revert before trying the next one.
            current_signs[idx] *= -1
            
        if best_flip_idx != -1:
            # Apply the best flip
            current_signs[best_flip_idx] *= -1
            current_val = best_flip_val
            improved = True
            
            if current_val < best_val:
                best_val = current_val
                best_signs = current_signs.copy()
                no_improve_count = 0
            else:
                # Improved local, but not global best? 
                # This shouldn't happen if we start from best. 
                # But if we are in a shake, we might have jumped to a higher valley.
                # We only update best_signs if strictly better.
                # If we improved locally but not globally, we continue from this new local state.
                pass
        else:
            no_improve_count += 1
            
    return list(best_signs)

# EVOLVE-BLOCK-END
