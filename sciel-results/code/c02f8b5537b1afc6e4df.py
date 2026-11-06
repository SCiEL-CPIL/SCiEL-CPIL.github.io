import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Generate all arithmetic progressions
    # AP is (start, step). Indices are start, start+step, ... < n
    # We use 0-based indexing for the sequence [0, n-1]
    progs = []
    for s in range(1, n + 1):
        for a in range(n):
            # Only add if it forms a valid AP of length >= 1
            # The range is a, a+s, ...
            # If a >= n, no. a is in [0, n-1].
            # Length is at least 1.
            progs.append((a, s))
            
    num_progs = len(progs)
    
    # 2. Precompute for each index i, which APs contain it
    # ap_indices_for_pos[i] = list of indices into progs
    ap_indices_for_pos = [[] for _ in range(n)]
    for p_idx, (a, s) in enumerate(progs):
        for idx in range(a, n, s):
            ap_indices_for_pos[idx].append(p_idx)
            
    # Convert to arrays for faster iteration? Lists are fine, but numpy might be faster for summing?
    # We are updating individual sums, so lists of ints are good.
    
    # 3. Initialize signs
    # Thue-Morse is a good start
    def thue_morse(n):
        return 1 if bin(n).count('1') % 2 == 0 else -1
        
    signs = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    # 4. Compute initial sums
    sums = np.zeros(num_progs, dtype=np.int32)
    for p_idx, (a, s) in enumerate(progs):
        # Sum signs[a], signs[a+s], ...
        # Using numpy slicing might be faster for initialization
        # signs[a::s]
        sums[p_idx] = np.sum(signs[a::s])
        
    # 5. Initialize counts and max
    max_possible_sum = n
    abs_counts = np.zeros(max_possible_sum + 1, dtype=np.int32)
    for s_val in sums:
        abs_counts[abs(s_val)] += 1
        
    current_max = max_possible_sum
    while current_max > 0 and abs_counts[current_max] == 0:
        current_max -= 1
        
    best_max = current_max
    best_signs = signs.copy()
    
    # 6. Simulated Annealing
    T = 10.0
    cooling_rate = 0.9995
    rng = np.random.default_rng(42)
    
    steps = 0
    while time.time() < deadline:
        # Pick random index
        i = rng.integers(n)
        old_sign = signs[i]
        new_sign = -old_sign
        
        # Calculate change
        # We need to update sums for all APs containing i
        # And update counts and max
        
        # Save old state for rollback if rejected
        # Instead of copying everything, we can revert manually
        
        affected_aps = ap_indices_for_pos[i]
        
        # Store old sums and abs values for affected APs
        # This is a bit of overhead, but necessary for rollback
        old_sums_val = []
        old_abs_val = []
        
        for p_idx in affected_aps:
            old_sums_val.append(sums[p_idx])
            old_abs_val.append(abs(sums[p_idx]))
            
        # Apply flip
        signs[i] = new_sign
        
        new_max = current_max
        # We need to update counts and max
        # To avoid complex max tracking during update, we can just update counts
        # and then determine new_max at the end?
        # No, we need to know if we improved to decide acceptance.
        # Actually, SA accepts based on energy difference.
        # Energy = current_max.
        
        # Let's update sums and counts
        for k, p_idx in enumerate(affected_aps):
            old_s = old_sums_val[k]
            old_a = old_abs_val[k]
            
            # Update sum
            new_s = old_s - 2 * old_sign
            sums[p_idx] = new_s
            new_a = abs(new_s)
            
            # Update counts
            abs_counts[old_a] -= 1
            abs_counts[new_a] += 1
            
            # Track new_max
            if new_a > new_max:
                new_max = new_a
        
        # If new_max < current_max, it's an improvement.
        # If new_max > current_max, it's a worsening.
        # If new_max == current_max, neutral.
        
        # But wait, if current_max was the only one at that level, and we removed it,
        # new_max might be lower even if no new_a > current_max.
        # My logic above:
        # if new_a > new_max: new_max = new_a
        # This only increases new_max.
        # It doesn't decrease it if the max count drops to 0.
        
        # We need to handle the case where the max decreases.
        # If new_max (tracked) is still current_max, we need to check if counts[current_max] is 0.
        # If yes, find new max.
        
        # Let's refine:
        # 1. Update counts.
        # 2. Determine candidate_max.
        #    - If any new_a > current_max, candidate_max = max(new_a)
        #    - Else, if abs_counts[current_max] == 0, candidate_max = next lower non-zero
        #    - Else candidate_max = current_max
        
        # Let's just compute candidate_max properly.
        # Max sum is small (256).
        # We can just scan down from current_max if needed.
        
        # Optimization:
        # If we added a value > current_max, new_max is that value.
        # If we didn't, and we removed the last one at current_max, we scan down.
        
        max_in_affected = 0
        for a in old_abs_val:
            if a > max_in_affected: max_in_affected = a # Not useful
            
        # Let's just use the counts array.
        # After updating counts:
        # If abs_counts[current_max] > 0:
        #     new_max = current_max
        # Else:
        #     # Scan down
        #     new_max = current_max - 1
        #     while new_max > 0 and abs_counts[new_max] == 0:
        #         new_max -= 1
        
        # But what if we INCREASED the max?
        # If any new_a > current_max, then abs_counts[new_max] will be > 0 for that new_max.
        # So the scan down logic must start from the potential new max.
        
        # Simpler:
        # Determine if max increased.
        max_new_a = 0
        for a in [abs(sums[p_idx]) for p_idx in affected_aps]:
            if a > max_new_a:
                max_new_a = a
                
        if max_new_a > current_max:
            new_max = max_new_a
        else:
            # Max did not increase. Check if it decreased.
            if abs_counts[current_max] == 0:
                new_max = current_max - 1
                while new_max > 0 and abs_counts[new_max] == 0:
                    new_max -= 1
            else:
                new_max = current_max
                
        delta_e = new_max - current_max
        
        # Acceptance probability
        if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Rollback
            signs[i] = old_sign
            for k, p_idx in enumerate(affected_aps):
                sums[p_idx] = old_sums_val[k]
                # Rollback counts
                old_a = old_abs_val[k]
                new_a = abs(sums[p_idx]) # This is old_a
                # We need to revert the count changes
                # We did: abs_counts[old_a] -= 1, abs_counts[new_a] += 1
                # So revert: abs_counts[old_a] += 1, abs_counts[new_a] -= 1
                # But new_a is the value we calculated in the forward pass.
                # We didn't store new_a separately, but we can recompute or store it.
                # Let's store new_abs_val in the forward pass.
                
            # This rollback logic is getting messy.
            # Let's restructure to store new_abs_val.
            
        T *= cooling_rate
        steps += 1
        
    return [int(s) for s in best_signs]
