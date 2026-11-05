"""Direct heuristic construction for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 256
    rng = random.Random(42)
    
    # 1. Precompute all APs
    # APs are defined by (start, step). 
    # We store the list of indices for each AP.
    # We also need to map index -> list of AP IDs containing it.
    
    ap_indices_list = []
    index_to_ap_ids = [[] for _ in range(n)]
    
    ap_id = 0
    for d in range(1, n):
        for s in range(n):
            # s is 0-indexed start. Corresponds to 1-indexed s+1.
            # AP: s, s+d, s+2d, ... < n
            indices = list(range(s, n, d))
            if not indices:
                continue
            # Store this AP
            ap_indices_list.append(indices)
            for idx in indices:
                index_to_ap_ids[idx].append(ap_id)
            ap_id += 1
            
    num_aps = ap_id
    
    # 2. Initial Signs
    # Try a few initializations and pick the best
    candidates = []
    
    # Random
    candidates.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # Thue-Morse
    signs_tm = [0] * n
    for i in range(n):
        t = 0
        x = i
        while x:
            t ^= (x & 1)
            x >>= 1
        signs_tm[i] = 1 if t == 0 else -1
    candidates.append(signs_tm)
    
    # Alternating
    candidates.append([1 if i % 2 == 0 else -1 for i in range(n)])
    
    # Another Random
    rng2 = random.Random(123)
    candidates.append([rng2.choice((-1, 1)) for _ in range(n)])
    
    best_signs = None
    best_disc = float('inf')
    
    # Helper to compute initial discrepancy and setup state
    def setup_state(signs):
        ap_sums = [0] * num_aps
        for ap_id, indices in enumerate(ap_indices_list):
            s = 0
            for idx in indices:
                s += signs[idx]
            ap_sums[ap_id] = s
        
        # Histogram of sums
        # Max possible sum is n (256). Min is -256.
        offset = n
        hist_size = 2 * n + 1
        hist = [0] * hist_size
        max_abs = 0
        for s in ap_sums:
            hist[s + offset] += 1
            if abs(s) > max_abs:
                max_abs = abs(s)
                
        return ap_sums, hist, max_abs

    for cand in candidates:
        ap_sums, hist, max_abs = setup_state(cand)
        if max_abs < best_disc:
            best_disc = max_abs
            best_signs = cand[:]
            best_ap_sums = ap_sums
            best_hist = hist

    signs = best_signs
    ap_sums = best_ap_sums
    hist = best_hist
    current_disc = best_disc
    
    offset = n
    hist_size = 2 * n + 1
    
    def get_max_from_hist(hist):
        max_abs = 0
        for val in hist:
            if val > 0:
                # We need to know the index to get the value
                pass
        # Better: scan outwards
        for k in range(offset, -1, -1):
            if hist[k] > 0 or hist[2*offset - k] > 0:
                return abs(k - offset)
        return 0

    current_disc = get_max_from_hist(hist)
    
    improved = True
    iter_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        improved = False
        
        # Random order of indices
        order = list(range(n))
        rng.shuffle(order)
        
        for i in order:
            if time.time() > deadline:
                break
            
            old_val = signs[i]
            new_val = -old_val
            signs[i] = new_val
            
            # Update APs containing i
            affected_aps = index_to_ap_ids[i]
            for ap_id in affected_aps:
                old_sum = ap_sums[ap_id]
                new_sum = old_sum - 2 * old_val # Since we flipped, change is -2*old_val
                ap_sums[ap_id] = new_sum
                
                # Update histogram
                hist[old_sum + offset] -= 1
                hist[new_sum + offset] += 1
            
            # Check new discrepancy
            new_disc = get_max_from_hist(hist)
            
            if new_disc < current_disc:
                current_disc = new_disc
                improved = True
            else:
                # Revert
                signs[i] = old_val
                for ap_id in affected_aps:
                    old_sum = ap_sums[ap_id]
                    new_sum = old_sum + 2 * old_val # Revert
                    ap_sums[ap_id] = new_sum
                    
                    hist[old_sum + offset] -= 1
                    hist[new_sum + offset] += 1
    
    return signs
# EVOLVE-BLOCK-END
