"""Program for low discrepancy on all arithmetic progressions.
Uses efficient discrete local search minimizing sum of squared discrepancies.
"""
import numpy as np
import time

def construct():
    n = 256
    
    # 1. Precompute Arithmetic Progressions
    # APs are defined by start s, step d.
    # We store each AP as a list of indices.
    aps = []
    pos_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n + 1):
        for s in range(0, n):
            # Generate AP: s, s+d, s+2d, ... < n
            current = s
            ap_indices = []
            while current < n:
                ap_indices.append(current)
                pos_to_aps[current].append(len(aps)) # Store index of this AP
                current += d
            if ap_indices:
                aps.append(ap_indices)
                
    num_aps = len(aps)
    
    # 2. Helper functions for state updates
    
    def compute_ap_sums(signs):
        """Compute sums for all APs given signs."""
        sums = np.zeros(num_aps, dtype=np.int32)
        for i, s in enumerate(signs):
            if s == 1:
                for ap_idx in pos_to_aps[i]:
                    sums[ap_idx] += 1
            else:
                for ap_idx in pos_to_aps[i]:
                    sums[ap_idx] -= 1
        return sums

    def get_max_abs(sums):
        return np.max(np.abs(sums))

    def get_sum_sq(sums):
        return np.sum(sums * sums)

    # 3. Local Search
    
    best_signs = None
    best_max_disc = float('inf')
    
    start_time = time.time()
    max_time = 55.0 # Leave 5s buffer
    
    # Multiple restarts to escape local minima
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        # Initialize randomly
        signs = np.random.choice([-1, 1], size=n)
        sums = compute_ap_sums(signs)
        sum_sq = get_sum_sq(sums)
        max_disc = get_max_abs(sums)
        
        if max_disc < best_max_disc:
            best_max_disc = max_disc
            best_signs = signs.copy()
            
        improved = True
        while improved and (time.time() - start_time < max_time):
            improved = False
            best_delta = 0
            best_pos = -1
            
            # Find the flip that reduces sum_sq the most
            for i in range(n):
                s_i = signs[i]
                ap_indices_for_i = pos_to_aps[i]
                if not ap_indices_for_i:
                    continue
                    
                sum_of_sums = 0
                for ap_idx in ap_indices_for_i:
                    sum_of_sums += sums[ap_idx]
                    
                delta = -4 * s_i * sum_of_sums + 4 * len(ap_indices_for_i)
                
                if delta < best_delta:
                    best_delta = delta
                    best_pos = i
            
            if best_pos != -1 and best_delta < 0:
                # Perform flip
                i = best_pos
                s_i = signs[i]
                signs[i] = -s_i
                
                # Update sums
                for ap_idx in pos_to_aps[i]:
                    sums[ap_idx] -= 2 * s_i
                    
                sum_sq += best_delta
                max_disc = get_max_abs(sums)
                improved = True
                
                if max_disc < best_max_disc:
                    best_max_disc = max_disc
                    best_signs = signs.copy()
                    
            else:
                # Stuck in local minimum for sum_sq
                # Try a random kick to escape
                kick_pos = np.random.randint(n)
                s_kick = signs[kick_pos]
                signs[kick_pos] = -s_kick
                for ap_idx in pos_to_aps[kick_pos]:
                    sums[ap_idx] -= 2 * s_kick
                sum_sq = get_sum_sq(sums) # Recalculate to be safe
                max_disc = get_max_abs(sums)
                
                if max_disc < best_max_disc:
                    best_max_disc = max_disc
                    best_signs = signs.copy()
                    
                # The while loop will exit because improved is still False
                # This effectively restarts the greedy search from a perturbed state
                pass 

    # Final check
    if best_signs is None:
        return [1] * n
        
    return [int(s) for s in best_signs]
