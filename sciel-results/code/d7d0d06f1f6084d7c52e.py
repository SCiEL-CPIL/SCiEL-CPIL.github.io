import numpy as np
import time
import random

def calc_discrepancy(s):
    """
    Calculate the maximum absolute signed sum over all arithmetic progressions.
    s: numpy array of -1, 1
    Returns: max_abs_sum
    """
    n = len(s)
    max_disc = 0
    # We can vectorize the inner loop over residues for a fixed d?
    # Or just loop. N=256 is small.
    
    for d in range(1, n):
        for r in range(d):
            sub = s[r::d]
            if len(sub) == 0:
                continue
            
            # c[0] = 0, c[k] = sum(sub[0..k-1])
            c = np.concatenate(([0], np.cumsum(sub)))
            
            # We want max_{0 <= j < k <= m} |c[k] - c[j]|
            # This is max( max(c[k] - min(c[0..k-1])), max(max(c[0..k-1]) - c[k]) )
            
            # Running min and max
            running_min = np.minimum.accumulate(c)
            running_max = np.maximum.accumulate(c)
            
            # min_prev[k] = min(c[0..k-1])
            # max_prev[k] = max(c[0..k-1])
            
            # Create arrays for min_prev and max_prev
            # min_prev[0] is unused (or inf)
            # min_prev[k] = running_min[k-1] for k >= 1
            
            # Efficient way:
            # pos_diffs = c - np.concatenate(([np.inf], running_min[:-1]))
            # neg_diffs = np.concatenate(([-np.inf], running_max[:-1])) - c
            
            # But creating these arrays is O(m).
            # Let's just use the slices.
            
            min_prev = running_min[:-1]
            max_prev = running_max[:-1]
            
            # c[1:] - min_prev
            pos = c[1:] - min_prev
            # max_prev - c[1:]
            neg = max_prev - c[1:]
            
            cur_max = max(pos.max(), neg.max())
            if cur_max > max_disc:
                max_disc = cur_max
                
    return max_disc

def find_worst_ap(s):
    """
    Find the arithmetic progression with the maximum absolute sum.
    Returns: (max_val, indices)
    """
    n = len(s)
    best_val = 0
    best_indices = None
    
    for d in range(1, n):
        for r in range(d):
            sub = s[r::d]
            if len(sub) == 0:
                continue
            
            c = np.concatenate(([0], np.cumsum(sub)))
            
            # Find k that maximizes |c[k] - c[j]| for j < k
            # We can find the pair (j, k)
            
            running_min = np.minimum.accumulate(c)
            running_max = np.maximum.accumulate(c)
            
            min_prev = running_min[:-1]
            max_prev = running_max[:-1]
            
            pos = c[1:] - min_prev
            neg = max_prev - c[1:]
            
            # Find max positive
            idx_pos = pos.argmax()
            val_pos = pos[idx_pos]
            # j is the index of the min in c[0..idx_pos]
            # min_prev[idx_pos] is that min value.
            # We need the index j where c[j] == min_prev[idx_pos]
            # And j < idx_pos + 1? No, min_prev[k] is min of c[0..k-1].
            # So for k = idx_pos + 1, j is argmin(c[0..idx_pos]).
            
            # Actually, we just need the indices in the original array.
            # The AP is defined by start index in sub, and length.
            # If the max is positive, it's c[k] - c[j] where k > j.
            # Sum is sub[j..k-1].
            # Start index in sub is j.
            # Length is k - j.
            
            if val_pos > best_val:
                best_val = val_pos
                k = idx_pos + 1
                j = np.argmin(c[:k]) # min in c[0..k-1]
                # Indices in s: r + j*d, ..., r + (k-1)*d
                start_idx = r + j * d
                length = k - j
                best_indices = list(range(start_idx, start_idx + length * d, d))
                
            if val_pos < 0: # Handle negative max (abs)
                 # Actually we take max(pos.max(), neg.max())
                 pass
            
            # Check negative
            idx_neg = neg.argmax()
            val_neg = neg[idx_neg]
            
            if val_neg > best_val:
                best_val = val_neg
                k = idx_neg + 1
                # max_prev[k-1] is max of c[0..k-1]
                # We want j such that c[j] = max_prev[k-1]
                j = np.argmax(c[:k])
                start_idx = r + j * d
                length = k - j
                best_indices = list(range(start_idx, start_idx + length * d, d))
                
    return best_val, best_indices

def construct():
    n = 256
    start_time = time.time()
    timeout = 55.0 # Safety margin
    
    best_signs = None
    best_disc = float('inf')
    
    # Initial random
    s = np.random.choice([-1, 1], size=n)
    d = calc_discrepancy(s)
    if d < best_disc:
        best_disc = d
        best_signs = s.copy()
        
    # Local Search
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() - start_time < timeout:
        # Find worst AP
        max_val, worst_indices = find_worst_ap(s)
        
        if max_val == 0:
            break
            
        improved = False
        
        # Try flipping each sign in the worst AP
        # Also try a few random flips if no improvement
        candidates = worst_indices.copy()
        
        # Shuffle to avoid bias
        random.shuffle(candidates)
        
        best_flip_idx = -1
        best_flip_disc = d
        
        for idx in candidates:
            s[idx] = -s[idx]
            new_d = calc_discrepancy(s)
            s[idx] = -s[idx] # revert
            
            if new_d < best_flip_disc:
                best_flip_disc = new_d
                best_flip_idx = idx
        
        if best_flip_idx != -1:
            s[best_flip_idx] = -s[best_flip_idx]
            d = best_flip_disc
            no_improve_count = 0
            if d < best_disc:
                best_disc = d
                best_signs = s.copy()
        else:
            # No improvement in worst AP, try random flip
            idx = random.randint(0, n-1)
            s[idx] = -s[idx]
            new_d = calc_discrepancy(s)
            
            if new_d < d:
                d = new_d
                no_improve_count = 0
                if d < best_disc:
                    best_disc = d
                    best_signs = s.copy()
            else:
                # Revert if worse? Or keep to escape?
                # Keep to escape local minima
                d = new_d
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                # Restart
                s = np.random.choice([-1, 1], size=n)
                d = calc_discrepancy(s)
                no_improve_count = 0
                if d < best_disc:
                    best_disc = d
                    best_signs = s.copy()

    return best_signs.tolist()

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We need to wrap it in the EVOLVE-BLOCK.
# EVOLVE-BLOCK-END
