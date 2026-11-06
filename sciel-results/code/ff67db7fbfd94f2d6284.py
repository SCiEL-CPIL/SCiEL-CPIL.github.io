# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    n = 16
    k = 4
    
    # Generate all 4-element subsets as bitmasks
    all_masks = []
    mask_to_idx = {}
    for idx, comb in enumerate(combinations(range(n), k)):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)
        mask_to_idx[mask] = idx
        
    N = len(all_masks)
    
    # Precompute pairwise intersections
    # inter[i][j] = all_masks[i] & all_masks[j]
    # This is a large matrix (1820 x 1820), but fits in memory (approx 3.3M ints)
    # Using a list of lists or a numpy array. Numpy might be faster for vectorized ops,
    # but we need specific pair checks. Let's use a flat array or list of lists.
    # 1820 * 1820 * 4 bytes (if int32) is ~13MB. Very safe.
    inter = [[0] * N for _ in range(N)]
    for i in range(N):
        mi = all_masks[i]
        for j in range(i + 1, N):
            mij = mi & all_masks[j]
            inter[i][j] = mij
            inter[j][i] = mij
        # Diagonal is the mask itself, but we only care about distinct sets i != j
        # Actually, for the check (a & b) == (a & s) == (b & s), we need:
        # inter[a][b] == inter[a][s] and inter[a][b] == inter[b][s]
        
    start_time = time.time()
    time_limit = 85.0  # 5s margin for 90s limit
    
    best_family = []
    best_size = 0
    
    # 1. Deterministic Greedy Baseline
    # This matches the "inspiration 1" behavior which scored 1.0
    # We use the precomputed intersection table for speed, though the simple version
    # was likely fast enough for the baseline.
    family = []
    for s in all_masks:
        conflict = False
        for i in range(len(family)):
            a = family[i]
            for j in range(i + 1, len(family)):
                b = family[j]
                # Check sunflower condition
                if inter[a][b] == inter[a][s] == inter[b][s]:
                    conflict = True
                    break
            if conflict:
                break
        
        if not conflict:
            family.append(s)
            
    if len(family) > best_size:
        best_size = len(family)
        best_family = family[:]
        
    # 2. Local Search / Simulated Annealing
    # Strategy: Start with the best family found so far.
    # Try to add new sets. If stuck, try swaps (remove one, add one or two).
    # We will use a randomized approach with a time limit.
    
    current_family = best_family[:]
    family_set = set(current_family)
    
    # Helper to check if a mask can be added to a list of masks
    # This is the bottleneck. We want to make it as fast as possible.
    # Using the precomputed inter table.
    
    def can_add(s, fam):
        # fam is a list of mask indices
        for i in range(len(fam)):
            a = fam[i]
            for j in range(i + 1, len(fam)):
                b = fam[j]
                if inter[a][b] == inter[a][s] == inter[b][s]:
                    return False
        return True

    # To speed up can_add, we can maintain a list of "conflict pairs" or similar?
    # Or just rely on the precomputed inter table which makes the inner loop very fast (list lookups).
    
    # Let's try a more aggressive local search:
    # While time remains:
    #   1. Try to add a random set not in family.
    #   2. If fail, try to swap: remove a random set from family, try to add a random set.
    #      If that fails, try to add two sets? (Too complex).
    #      If swap succeeds, accept it.
    #   3. Periodically, try to "repair" by removing a random set and greedily filling the gap?
    
    # Actually, a simpler and effective heuristic for these problems is "Randomized Greedy with Restarts"
    # combined with "Local Improvement".
    # The current program does restarts. Let's stick to that but optimize the check.
    
    # The precomputed inter table should make the check very fast.
    # Let's run multiple randomized greedy restarts.
    
    all_indices = list(range(N))
    
    while time.time() - start_time < time_limit:
        random.shuffle(all_indices)
        family = []
        
        for s_idx in all_indices:
            if len(family) > best_size:
                # Check time periodically
                if time.time() - start_time > time_limit:
                    break
            conflict = False
            for i in range(len(family)):
                a = family[i]
                for j in range(i + 1, len(family)):
                    b = family[j]
                    if inter[a][b] == inter[a][s_idx] == inter[b][s_idx]:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                family.append(s_idx)
                
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # Try local search on the current best
        # Pick a random set in the family, remove it, and try to add a new set.
        # This is complex. Let's stick to greedy restarts for now, but maybe with a twist.
        # What if we try to add sets in a specific order that prioritizes those with fewer conflicts?
        # Or just random restarts are enough.
        
        # Let's add a simple local search step after each restart:
        # Try to swap out one element for another to allow more additions?
        # For now, just restarts.
        
    # Convert best_family (indices) to lists of integers
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
