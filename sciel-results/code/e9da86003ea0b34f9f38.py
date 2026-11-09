import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin before 60s kill

    # 1. Precompute APs
    # An AP is defined by start s (0-indexed) and step d.
    # Indices: s, s+d, ..., < n
    ap_indices = []
    # We need to map index i to list of AP indices containing i
    # impact[i] = list of ap_idx
    
    # First pass: collect all APs
    # To ensure we cover all "arithmetic progressions contained in [1, 256]",
    # we iterate over all possible steps d and starts s.
    # Note: The problem says "contained in [1, 256]". 
    # Usually this means the indices are within 0..n-1.
    # "any start, positive step and length including one"
    
    for d in range(1, n + 1):
        for s in range(n):
            # Generate indices
            indices = list(range(s, n, d))
            if indices:
                ap_indices.append(indices)
                
    num_aps = len(ap_indices)
    
    # Build impact map: for each index i, which APs contain it?
    impact = [[] for _ in range(n)]
    for ap_idx, indices in enumerate(ap_indices):
        for i in indices:
            impact[i].append(ap_idx)

    # 2. Initialization
    rng = random.Random(42)
    
    # Start with Thue-Morse
    def thue_morse(n):
        return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
        
    signs = thue_morse(n)
    
    # Calculate initial sums
    ap_sums = [0] * num_aps
    for ap_idx, indices in enumerate(ap_indices):
        s = 0
        for i in indices:
            s += signs[i]
        ap_sums[ap_idx] = s
            
    # Build histogram of absolute sums
    # Max possible sum is n (256)
    max_possible = n
    hist = [0] * (max_possible + 1)
    for s in ap_sums:
        hist[abs(s)] += 1
        
    # Find current max discrepancy
    current_max = 0
    for j in range(max_possible, -1, -1):
        if hist[j] > 0:
            current_max = j
            break
            
    best_signs = signs[:]
    best_disc = current_max
    
    # 3. Simulated Annealing
    temperature = 10.0
    cooling_rate = 0.999
    min_temp = 0.1
    
    # Helper to find max from hist, starting from a specific range
    def get_max_from_hist(hist, start_val, end_val):
        # We only need to check a small range around current_max
        # But if we accept a bad move, max might jump.
        # If we reject, we revert.
        # So we just need to find the new max after a move.
        # The max can only change by at most 2 (flip changes sum by 2).
        # So we check from current_max + 2 down to current_max - 2.
        # But if we are doing random walks, max might stay same.
        # Actually, if we flip, sums change by 2.
        # So new abs sums are old_abs +/- 2 or same?
        # |S| -> |S+2| or |S-2|.
        # If S=5, |5|->|7|=7 or |3|=3.
        # So max can increase by 2 or decrease by 2.
        
        # To be safe and simple, we can just scan the whole hist if it's small?
        # Hist size is 257. Scanning 257 items is very fast.
        for j in range(max_possible, -1, -1):
            if hist[j] > 0:
                return j
        return 0

    while time.time() - start_time < time_limit:
        # Choose move
        # 70% chance: Pick index from a worst AP
        # 30% chance: Random index
        
        if rng.random() < 0.7:
            # Find an AP with current_max
            # We can pick a random one from those with max disc
            # To avoid scanning all APs every time, we could maintain a list,
            # but scanning 1000 APs is fast enough.
            worst_aps = [k for k in range(num_aps) if abs(ap_sums[k]) == current_max]
            if not worst_aps:
                # Should not happen if current_max is correct
                idx = rng.randint(0, n - 1)
            else:
                target_ap = rng.choice(worst_aps)
                target_indices = ap_indices[target_ap]
                target_sum = ap_sums[target_ap]
                
                # Choose candidate indices in this AP that would reduce |sum|
                candidates = []
                if target_sum > 0:
                    # Flip +1 to -1
                    for i in target_indices:
                        if signs[i] == 1:
                            candidates.append(i)
                elif target_sum < 0:
                    # Flip -1 to +1
                    for i in target_indices:
                        if signs[i] == -1:
                            candidates.append(i)
                else:
                    # Sum is 0, flip anything
                    candidates = target_indices[:]
                
                if candidates:
                    idx = rng.choice(candidates)
                else:
                    # No candidates (e.g. sum > 0 but all are -1? Impossible)
                    idx = rng.randint(0, n - 1)
        else:
            idx = rng.randint(0, n - 1)
            
        # Perform flip
        old_sign = signs[idx]
        new_sign = -old_sign
        delta = 2 * new_sign # Change in sum for each AP containing idx is 2*new_sign - 2*old_sign = 4*new_sign?
        # Wait.
        # Sum = ... + signs[idx] + ...
        # New Sum = ... + new_sign + ...
        # Delta = new_sign - old_sign = -old_sign - old_sign = -2 * old_sign.
        # If old_sign=1, new=-1, delta=-2.
        # If old_sign=-1, new=1, delta=2.
        
        delta_sum = -2 * old_sign
        
        # Save old state for impact set
        affected_aps = impact[idx]
        old_sums = {}
        old_hist = hist[:] # Copy hist
        
        # Apply change
        for k in affected_aps:
            old_s = ap_sums[k]
            old_sums[k] = old_s
            old_abs = abs(old_s)
            hist[old_abs] -= 1
            
            new_s = old_s + delta_sum
            ap_sums[k] = new_s
            new_abs = abs(new_s)
            hist[new_abs] += 1
            
        # Find new max
        new_max = get_max_from_hist(hist, 0, max_possible)
        
        delta_disc = new_max - current_max
        
        # Acceptance
        accepted = False
        if delta_disc < 0:
            accepted = True
        elif delta_disc == 0:
            accepted = True
        else:
            prob = math.exp(-delta_disc / temperature)
            if rng.random() < prob:
                accepted = True
                
        if accepted:
            signs[idx] = new_sign
            current_max = new_max
            if current_max < best_disc:
                best_disc = current_max
                best_signs = signs[:]
        else:
            # Revert
            signs[idx] = old_sign
            hist = old_hist
            for k in affected_aps:
                ap_sums[k] = old_sums[k]
                
        # Cooling
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp

    return best_signs
