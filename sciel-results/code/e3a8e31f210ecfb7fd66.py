import random
import time
from itertools import combinations

def construct():
    # 1. Generate all 4-subsets of [1, 16] as bitmasks
    n = 16
    all_sets = []
    for combo in combinations(range(n), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    num_sets = len(all_sets)
    
    # Precompute intersections for all pairs? No, too big (1820^2 ~ 3M).
    # We will compute on the fly or use the dynamic structure.
    
    def check_valid(chosen_indices, c_idx):
        """
        Checks if adding c_idx to chosen_indices forms a sunflower.
        chosen_indices is a list of indices into all_sets.
        We maintain a helper structure to speed this up.
        """
        c_mask = all_sets[c_idx]
        
        # For each A in chosen, let M = A & C.
        # We need to know if there is a B in chosen such that A & B == M and B & C == M.
        
        # We can maintain a dict: pair_intersections
        # Or simpler: for each A, a dict of intersection -> list of B indices.
        # But building this from scratch each time is O(N^2).
        # We can pass the structure as an argument and update it.
        
        # Let's use a simpler O(N^2) check first, but optimized with bit operations.
        # N is max ~250. N^2 is 62500. 1820 * 62500 is ~113M operations.
        # In Python, this might be slow (10-20s per restart).
        # We need to be faster.
        
        # Let's use the dynamic structure approach.
        # The caller will maintain `inter_struct`.
        # `inter_struct[a_idx]` is a dict: mask -> list of b_idx.
        
        # Actually, let's just do the O(N^2) check but with early termination and bitmasks.
        # 250^2 = 62,500 checks.
        # Each check is 3 bitwise ANDs and comparisons.
        # 1820 candidates * 62,500 = 113,750,000 operations.
        # Python can do ~10-50M simple ops/sec. So ~2-10s per full pass.
        # We have 90s budget. So ~10-40 passes.
        # This is feasible.
        
        for i in range(len(chosen_indices)):
            a_idx = chosen_indices[i]
            a_mask = all_sets[a_idx]
            ac_mask = a_mask & c_mask
            
            for j in range(i+1, len(chosen_indices)):
                b_idx = chosen_indices[j]
                b_mask = all_sets[b_idx]
                
                # Check if A & B == AC and B & C == AC
                if (a_mask & b_mask) == ac_mask and (b_mask & c_mask) == ac_mask:
                    return False
        return True

    best_size = 0
    best_family = []
    start_time = time.time()
    time_limit = 85.0 # Safety margin
    
    # We will run multiple randomized greedy passes
    # Shuffle the list of all set indices
    indices = list(range(num_sets))
    
    for restart in range(200): # Max restarts, break on time
        if time.time() - start_time > time_limit:
            break
            
        random.shuffle(indices)
        chosen = []
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            # Check if idx can be added
            # Optimization: if len(chosen) is large, check is expensive.
            # We can skip if we are already smaller than best? No, we want to build.
            
            if check_valid(chosen, idx):
                chosen.append(idx)
                
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
    # Convert best_family indices back to sorted lists of elements
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elements = [i+1 for i in range(n) if (mask >> i) & 1]
        result.append(elements)
        
    return result
