"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_ap_masks(n):
    """
    Precompute for each index i, the list of (mask, length) tuples for all
    arithmetic progressions that include i.
    
    For flipping bit i, we need to know which APs are affected.
    An AP is defined by (start, step, length) where all indices are in [0, n).
    The sum of an AP = sum of signs at those indices.
    When we flip sign at index i, the sum changes by +2 or -2 depending on current sign.
    
    Actually, for the "discrepancy" we track the maximum absolute sum over all APs.
    We can maintain a dictionary: ap_id -> current_sum, and track the max abs sum.
    But that's complex. Instead, let's precompute for each index i, the list of
    (ap_index, position_in_ap) or just the list of ap_indices that contain i.
    
    Then we maintain an array of sums for each AP, and a data structure to find max abs.
    
    Simpler approach: precompute for each index i, the list of APs containing i.
    Each AP is identified by an integer id. We maintain sums[ap_id] and update it
    when we flip a bit.
    
    Number of APs: For each step d in [1, n], for each start r in [0, min(d, n)-1],
    the AP is r, r+d, r+2d, ... The length is floor((n-1-r)/d) + 1.
    Total APs = sum_{d=1}^{n} min(d, n) * (average length)... Actually, total number
    of distinct APs (including singletons) is sum_{d=1}^{n} min(d, n) = sum_{d=1}^{n} d = n(n+1)/2.
    For n=256, that's 256*257/2 = 32896 APs.
    
    Each AP can be stored as a bitmask (256 bits = 4 uint64 words).
    For each index i, we store the list of AP ids that contain i.
    """
    n = 256
    # Map from ap_id to (mask_tuple, length)
    # But we need to update sums quickly.
    
    # Alternative: Don't use bitmasks for the APs themselves.
    # Instead, for each index i, precompute the list of ap_ids that contain i.
    # Then maintain an array `ap_sums` of size num_aps.
    # When flipping index i, for each ap_id in aps_containing_i[i], update ap_sums[ap_id] by +/-2.
    # Then we need to find max |ap_sums|. This is O(num_aps) per flip, which is 32896 * 80000 flips = too slow.
    
    # Better: maintain a count of how many APs have each absolute sum value?
    # The sums range from -256 to 256, so at most 513 distinct values.
    # We can maintain a frequency array: freq[v] = number of APs with sum = v.
    # Then max abs sum is found by scanning from 256 down to 1.
    
    # Let's go with the frequency approach.
    
    # Build ap_sums and the mapping from index to list of ap_ids.
    ap_sums = np.zeros(0, dtype=np.int32)  # placeholder
    idx_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    ap_lengths = []
    
    for d in range(1, n + 1):
        for r in range(min(d, n)):
            if r >= n:
                break
            # AP: r, r+d, r+2d, ...
            indices = list(range(r, n, d))
            if not indices:
                continue
            length = len(indices)
            ap_lengths.append(length)
            
            # Add this AP to each index in the AP
            for idx in indices:
                idx_to_aps[idx].append(ap_id)
            
            ap_id += 1
    
    num_aps = ap_id
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    
    # Initialize ap_sums with all +1 signs
    for ap_id_idx in range(num_aps):
        # We need to know which indices are in this AP
        # We didn't store the indices per AP. Let's rebuild.
        pass
    
    # This is getting complex. Let me restructure.
    
    # Better: store for each AP, the list of indices.
    # Then initialize sums and build the reverse mapping.
    
    # Let me restart the approach more cleanly.
    
    return None  # placeholder

def _build_ap_data(n):
    """
    Precompute data for fast discrepancy evaluation.
    Returns:
        ap_indices: list of lists, ap_indices[k] = list of indices in AP k
        idx_to_aps: list of lists, idx_to_aps[i] = list of AP ids containing index i
        num_aps: total number of APs
    """
    n = 256
    ap_indices = []
    idx_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n + 1):
        for r in range(min(d, n)):
            if r >= n:
                break
            indices = list(range(r, n, d))
            if not indices:
                continue
            ap_id = len(ap_indices)
            ap_indices.append(indices)
            for idx in indices:
                idx_to_aps[idx].append(ap_id)
    
    num_aps = len(ap_indices)
    return ap_indices, idx_to_aps, num_aps

def _init_sums(signs, ap_indices):
    """Initialize AP sums given signs."""
    n = len(signs)
    num_aps = len(ap_indices)
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for k, indices in enumerate(ap_indices):
        s = 0
        for idx in indices:
            s += signs[idx]
        ap_sums[k] = s
    return ap_sums

def _max_abs_from_sums(ap_sums):
    """Get max absolute sum from ap_sums array."""
    if len(ap_sums) == 0:
        return 1
    return int(np.max(np.abs(ap_sums)))

def _update_freq(ap_sums, old_val, new_val, freq):
    """Update frequency array: decrement old_val, increment new_val."""
    freq[old_val + 256] -= 1
    freq[new_val + 256] += 1

def _find_max_abs(freq):
    """Find max absolute value from frequency array. freq is indexed by val+256."""
    for v in range(256, -1, -1):
        if v > 0 and (freq[v + 256] > 0 or freq[-v + 256] > 0):
            return v
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s budget, 10s safety margin
    
    n = 256
    rng = random.Random(42)
    
    # Precompute AP data
    ap_indices, idx_to_aps, num_aps = _build_ap_data(n)
    
    # Initialize with random signs
    signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)
    
    # Initialize AP sums and frequency
    ap_sums = _init_sums(signs, ap_indices)
    max_disc = _max_abs_from_sums(ap_sums)
    
    # Build frequency array
    # Values range from -n to n, so we offset by n
    offset = n
    freq = np.zeros(2 * n + 1, dtype=np.int32)
    for s in ap_sums:
        freq[s + offset] += 1
    
    best_signs = signs.copy()
    best_score = max_disc
    current_score = max_disc
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    alpha = 0.9995
    
    while time.time() < deadline:
        idx = rng.randint(0, n - 1)
        
        # Current sign at idx
        old_sign = signs[idx]
        new_sign = -old_sign
        
        # Update AP sums for all APs containing idx
        affected_aps = idx_to_aps[idx]
        
        old_sums = []
        for ap_id in affected_aps:
            old_sums.append(ap_sums[ap_id])
        
        # Apply flip
        delta = 2 * new_sign  # +2 if new_sign=1, -2 if new_sign=-1
        # Actually, the change in sum for each AP is: new_sum = old_sum - old_sign + new_sign = old_sum + 2*new_sign
        # Since new_sign = -old_sign, change = -2*old_sign
        change = -2 * old_sign
        
        for ap_id in affected_aps:
            new_sum = ap_sums[ap_id] + change
            _update_freq(ap_sums, ap_sums[ap_id], new_sum, freq)
            ap_sums[ap_id] = new_sum
        
        signs[idx] = new_sign
        
        new_score = _find_max_abs(freq)
        
        # Acceptance
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_signs = signs.copy()
        else:
            delta_score = new_score - current_score
            if delta_score == 0:
                # Accept equal moves to explore
                current_score = new_score
            elif delta_score > 0:
                prob = np.exp(-delta_score / T)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    signs[idx] = old_sign
                    revert_change = 2 * old_sign
                    for ap_id in affected_aps:
                        new_sum = ap_sums[ap_id] + revert_change
                        _update_freq(ap_sums, ap_sums[ap_id], new_sum, freq)
                        ap_sums[ap_id] = new_sum
            # else: delta_score < 0 already handled above
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_signs.tolist()

# EVOLVE-BLOCK-END
