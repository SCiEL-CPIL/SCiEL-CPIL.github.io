import time
import random
import numpy as np
import math

def _compute_all_ap_discrepancies(signs):
    """
    Compute the maximum absolute subarray sum for all arithmetic progressions.
    Returns the global maximum discrepancy.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 1
    
    # We iterate over step d and remainder r
    # For a fixed d, the APs are disjoint sets of indices? No, they are interleaved.
    # Actually, for fixed d, the APs are:
    # r, r+d, r+2d, ...
    # These are disjoint for fixed d.
    
    # Optimization: Group by d.
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                continue
            indices = np.arange(r, n, d)
            if len(indices) <= 1:
                continue
            sub = signs_arr[indices]
            # Compute cumulative sum
            cum = np.concatenate(([0], np.cumsum(sub)))
            # Max subarray sum (Kadane's variant for max/min)
            # Max subarray sum = max(cum[j] - min(cum[i]) for i < j)
            # We can compute this efficiently.
            
            # min_prefix[i] = min(cum[0..i])
            # max_prefix[i] = max(cum[0..i])
            
            # To find max subarray sum:
            # max_{j} (cum[j] - min_{i<j} cum[i])
            # Let min_cum_before[j] = min(cum[0..j-1])
            # Then max_sum = max_j (cum[j] - min_cum_before[j])
            
            # Similarly for min subarray sum:
            # min_{j} (cum[j] - max_{i<j} cum[i])
            
            # Vectorized approach:
            # cum has length m+1
            # min_cum_before has length m+1, where min_cum_before[0]=0, min_cum_before[j]=min(cum[0..j-1])
            # This is just minimum.accumulate(cum) shifted?
            # min_cum_before[1:] = np.minimum.accumulate(cum[:-1])
            
            min_cum_before = np.empty_like(cum)
            min_cum_before[0] = 0
            if len(cum) > 1:
                min_cum_before[1:] = np.minimum.accumulate(cum[:-1])
                
            max_cum_before = np.empty_like(cum)
            max_cum_before[0] = 0
            if len(cum) > 1:
                max_cum_before[1:] = np.maximum.accumulate(cum[:-1])
                
            # Max sum ending at j is cum[j] - min_cum_before[j]
            max_sums = cum - min_cum_before
            min_sums = cum - max_cum_before
            
            current_max = int(np.max(max_sums))
            current_min = int(np.min(min_sums))
            
            disc = max(abs(current_max), abs(current_min))
            if disc > max_abs:
                max_abs = disc
                
    return max_abs

def _build_ap_structure(n):
    """
    Pre-compute the list of APs and their membership for faster local search updates.
    Returns:
        aps: list of lists of indices
        ap_membership: list of lists of ap_indices for each index in [0..n-1]
    """
    aps = []
    ap_membership = [[] for _ in range(n)]
    
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

def _update_ap_discrepancy(ap_indices, signs_arr, ap_idx, max_sums, min_sums):
    """
    Update the max and min subarray sum for a specific AP.
    """
    sub = signs_arr[ap_indices]
    cum = np.concatenate(([0], np.cumsum(sub)))
    
    min_cum_before = np.empty_like(cum)
    min_cum_before[0] = 0
    if len(cum) > 1:
        min_cum_before[1:] = np.minimum.accumulate(cum[:-1])
        
    max_cum_before = np.empty_like(cum)
    max_cum_before[0] = 0
    if len(cum) > 1:
        max_cum_before[1:] = np.maximum.accumulate(cum[:-1])
        
    max_sums[ap_idx] = np.max(cum - min_cum_before)
    min_sums[ap_idx] = np.min(cum - max_cum_before)

def _get_global_disc(max_sums, min_sums):
    """
    Calculate the global maximum discrepancy from per-AP max/min sums.
    """
    if len(max_sums) == 0:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def _find_best_flip(aps, ap_membership, signs, max_sums, min_sums, n, rng, top_k=10):
    """
    Find the best single bit flip to reduce the global discrepancy.
    We focus on the top_k worst APs.
    """
    n_aps = len(aps)
    if n_aps == 0:
        return -1, 1
    
    # Identify the top_k worst APs based on current discrepancy
    # Disc for AP i is max(abs(max_sums[i]), abs(min_sums[i]))
    discs = np.maximum(np.abs(max_sums), np.abs(min_sums))
    
    # Get indices of top_k worst APs
    if n_aps <= top_k:
        worst_aps = np.arange(n_aps)
    else:
        # Use argpartition for efficiency
        worst_aps = np.argpartition(discs, -top_k)[-top_k:]
        
    # Collect candidate indices from these worst APs
    candidates = set()
    for ap_idx in worst_aps:
        for idx in aps[ap_idx]:
            candidates.add(idx)
            
    if not candidates:
        return -1, 1
        
    signs_arr = np.asarray(signs, dtype=np.int8)
    best_idx = -1
    best_new_disc = _get_global_disc(max_sums, min_sums)
    
    # Shuffle candidates to break ties randomly
    cand_list = list(candidates)
    rng.shuffle(cand_list)
    
    for idx in cand_list:
        # Try flipping idx
        old_val = signs_arr[idx]
        signs_arr[idx] = -old_val
        
        # Update all APs containing idx
        for ap_idx in ap_membership[idx]:
            _update_ap_discrepancy(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
            
        new_disc = _get_global_disc(max_sums, min_sums)
        
        if new_disc < best_new_disc:
            best_new_disc = new_disc
            best_idx = idx
            # We can't break immediately because we need to revert the flip for the next candidate
            # But we can keep track of the best state?
            # Actually, we must revert the flip to test the next candidate.
            
        # Revert flip
        signs_arr[idx] = old_val
        for ap_idx in ap_membership[idx]:
            _update_ap_discrepancy(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
            
    return best_idx, best_new_disc

def construct():
    start_time = time.time()
    # 90s limit, leave 5s margin
    deadline = start_time + 85
    
    n = 256
    rng = random.Random(42)
    
    # Build AP structure
    aps, ap_membership = _build_ap_structure(n)
    
    # Initialization: Thue-Morse
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    # Initialize max_sums and min_sums
    n_aps = len(aps)
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    
    for ap_idx, ap in enumerate(aps):
        _update_ap_discrepancy(ap, signs_arr, ap_idx, max_sums, min_sums)
        
    current_disc = _get_global_disc(max_sums, min_sums)
    best_disc = current_disc
    best_signs = signs[:]
    
    # Phase 1: Greedy Local Search
    # Repeatedly find the best single flip
    while time.time() < deadline:
        # Check for improvement
        flip_idx, new_disc = _find_best_flip(aps, ap_membership, signs, max_sums, min_sums, n, rng)
        
        if flip_idx == -1:
            # No improvement found
            break
            
        # Apply the best flip
        signs[flip_idx] = -signs[flip_idx]
        signs_arr[flip_idx] = -signs_arr[flip_idx]
        for ap_idx in ap_membership[flip_idx]:
            _update_ap_discrepancy(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
            
        current_disc = new_disc
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs[:]
            
        if best_disc == 1:
            return best_signs
            
    # Phase 2: Simulated Annealing
    # If greedy got stuck, use SA to escape local minima
    if time.time() < deadline - 5:
        temp = 2.0
        cool_rate = 0.999
        no_improve = 0
        max_no_improve = 500
        
        # Reset signs to best found so far
        signs = best_signs[:]
        signs_arr = np.asarray(signs, dtype=np.int8)
        for ap_idx, ap in enumerate(aps):
            _update_ap_discrepancy(ap, signs_arr, ap_idx, max_sums, min_sums)
        current_disc = best_disc
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            idx = rng.randint(0, n - 1)
            old_val = signs[idx]
            new_val = -old_val
            
            signs[idx] = new_val
            signs_arr[idx] = new_val
            
            for ap_idx in ap_membership[idx]:
                _update_ap_discrepancy(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
                
            new_disc = _get_global_disc(max_sums, min_sums)
            
            delta = new_disc - current_disc
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 0.01)):
                current_disc = new_disc
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                signs[idx] = old_val
                signs_arr[idx] = old_val
                for ap_idx in ap_membership[idx]:
                    _update_ap_discrepancy(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
                no_improve += 1
                
            temp *= cool_rate
            
            if best_disc == 1:
                return best_signs
                
        # Phase 3: Final Polish
        if time.time() < deadline - 2:
            signs = best_signs[:]
            signs_arr = np.asarray(signs, dtype=np.int8)
            for ap_idx, ap in enumerate(aps):
                _update_ap_discrepancy(ap, signs_arr, ap_idx, max_sums, min_sums)
            current_disc = best_disc
            
            # Run greedy again
            while time.time() < deadline - 2:
                flip_idx, new_disc = _find_best_flip(aps, ap_membership, signs, max_sums, min_sums, n, rng)
                if flip_idx == -1:
                    break
                signs[flip_idx] = -signs[flip_idx]
                signs_arr[flip_idx] = -signs_arr[flip_idx]
                for ap_idx in ap_membership[flip_idx]:
                    _update_ap_discrepancy(aps[ap_idx], signs_arr, ap_idx, max_sums, min_sums)
                current_disc = new_disc
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
                if best_disc == 1:
                    return best_signs

    return best_signs

# Verification function for debugging (not used in construct)
def _verify(signs):
    return _compute_all_ap_discrepancies(signs)
