"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _get_aps():
    """Get list of APs as index arrays."""
    aps = []
    for step in range(1, N // 2 + 1):
        for start in range(1, N - step + 1):
            max_len = (N - start) // step + 1
            if max_len < 2:
                continue
            indices = [start - 1 + k * step for k in range(max_len)]
            for length in range(2, max_len + 1):
                aps.append(np.array(indices[:length], dtype=np.int32))
    return aps

def _precompute_membership(aps):
    """Precompute membership matrix and initial sums."""
    M = len(aps)
    # Create a flat array of all indices in all APs
    total_len = sum(len(ap) for ap in aps)
    flat_indices = np.empty(total_len, dtype=np.int32)
    # Also need to know for each position, which APs it's in, or just update sums directly.
    # Since we need to update sums for ALL APs containing pos i, we need the inverse mapping or just recompute.
    # Better: Precompute for each position i, the list of AP indices containing it.
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(ap_idx)
    
    # Convert to numpy arrays for faster iteration if needed, but lists are fine for small counts.
    # Actually, for vectorized update:
    # When flipping pos i, new_sum[j] = old_sum[j] + 2 * signs[i] for all j in pos_to_aps[i].
    # We can store pos_to_aps as a list of arrays.
    pos_to_aps_arrays = [np.array(lst, dtype=np.int32) for lst in pos_to_aps]
    
    return pos_to_aps_arrays

def _compute_initial_sums(signs, aps):
    """Compute initial sums for all APs."""
    sums = np.zeros(len(aps), dtype=np.int32)
    for i, ap in enumerate(aps):
        sums[i] = np.sum(signs[ap])
    return sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit, keep 5s margin
    
    # 1. Precomputation
    aps = _get_aps()
    pos_to_aps = _precompute_membership(aps)
    
    # 2. Initialize with a good candidate
    # Thue-Morse is often good for discrepancy problems
    signs = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(N)], dtype=np.int32)
    
    # Evaluate initial
    sums = _compute_initial_sums(signs, aps)
    current_max = np.max(np.abs(sums))
    best_signs = signs.copy()
    best_val = current_max
    
    # 3. Simulated Annealing
    # We need to be very fast.
    # Flip position i:
    #   signs[i] *= -1
    #   delta = 2 * old_signs[i]
    #   sums[pos_to_aps[i]] += delta
    #   new_max = max(abs(sums)) -- this is the bottleneck if done naively.
    #   We can maintain a histogram of |sums| to get max in O(1) or O(log N).
    
    # Let's use a histogram approach for max tracking.
    # Sums can range from -256 to 256. So |sums| is 0..256.
    max_possible_sum = N
    hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
    
    def update_hist(sums):
        """Rebuild histogram from sums array."""
        hist[:] = 0
        abs_sums = np.abs(sums)
        np.add.at(hist, abs_sums, 1)
        return hist
    
    def get_max_from_hist(hist):
        # Find largest index with non-zero count
        # hist is small (257 elements)
        for i in range(len(hist) - 1, -1, -1):
            if hist[i] > 0:
                return i
        return 0

    hist = update_hist(sums)
    current_max = get_max_from_hist(hist)
    best_val = current_max
    best_signs = signs.copy()
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.9995
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 2.0:
        if T < 0.01:
            # Try to restart or just stop?
            # Let's do a few more greedy steps or break.
            T = 0.01 
            
        # Pick a random position to flip
        pos = rng.integers(0, N)
        
        old_sign = signs[pos]
        new_sign = -old_sign
        delta = 2 * old_sign  # Change in sum for APs containing pos
        
        # Apply flip
        signs[pos] = new_sign
        
        # Update sums
        affected_aps = pos_to_aps[pos]
        if len(affected_aps) > 0:
            # Update histogram for affected APs
            old_abs_sums = np.abs(sums[affected_aps])
            new_sums_vals = sums[affected_aps] + delta
            new_abs_sums = np.abs(new_sums_vals)
            
            # Decrement old counts
            np.add.at(hist, old_abs_sums, -1)
            # Increment new counts
            np.add.at(hist, new_abs_sums, 1)
            
            # Update sums array
            sums[affected_aps] = new_sums_vals
            
        new_max = get_max_from_hist(hist)
        
        # Acceptance criterion
        if new_max < current_max:
            current_max = new_max
            if current_max < best_val:
                best_val = current_max
                best_signs = signs.copy()
        else:
            # Delta E = new_max - current_max (>= 0)
            delta_e = new_max - current_max
            if delta_e == 0:
                accept = True
            else:
                # Metropolis
                # exp(-delta_e / T)
                # If T is very small, this is 0.
                # To avoid underflow/overflow, we can just check if rng.random() < exp(...)
                # But exp(-large) is 0.
                # Since delta_e is integer >= 1, and T starts at 10, it's fine.
                try:
                    prob = np.exp(-delta_e / T)
                except:
                    prob = 0.0
                accept = rng.random() < prob
                
            if not accept:
                # Revert
                signs[pos] = old_sign
                if len(affected_aps) > 0:
                    # Revert histogram
                    old_abs_sums_revert = np.abs(sums[affected_aps]) # These are the "new" ones
                    # Wait, sums[affected_aps] were updated.
                    # We need to revert sums first to calculate old abs?
                    # No, we have old_abs_sums from before update?
                    # Let's just recompute the revert carefully.
                    
                    # Current state: sums[affected_aps] = new_sums_vals
                    # We want to go back to old_sums_vals = new_sums_vals - delta
                    revert_sums = sums[affected_aps] - delta
                    revert_abs = np.abs(revert_sums)
                    current_abs = np.abs(sums[affected_aps])
                    
                    np.add.at(hist, current_abs, -1)
                    np.add.at(hist, revert_abs, 1)
                    
                    sums[affected_aps] = revert_sums
                    
        T *= cooling_rate
        
    # Final verification (optional, but good for safety)
    # return best_signs.tolist()
    
    # Ensure return type is list of Python integers
    return best_signs.tolist()

# EVOLVE-BLOCK-END
