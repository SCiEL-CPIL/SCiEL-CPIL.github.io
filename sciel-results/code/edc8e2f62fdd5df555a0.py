import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 16
    k = 4
    
    # Precompute all 4-subsets as bitmasks for fast intersection
    all_subsets = list(combinations(range(1, n + 1), k))
    masks = []
    for s in all_subsets:
        m = 0
        for bit in s:
            m |= (1 << (bit - 1))
        masks.append(m)
    
    num_sets = len(all_subsets)
    
    # Precompute pairwise intersections between all sets
    # inter[i][j] = masks[i] & masks[j]
    # This is 1820 x 1820 = ~3.3M entries, each an int
    # Memory: ~26MB, acceptable
    inter = [[0] * num_sets for _ in range(num_sets)]
    for i in range(num_sets):
        mi = masks[i]
        for j in range(i + 1, num_sets):
            v = mi & masks[j]
            inter[i][j] = v
            inter[j][i] = v
    
    def can_add(family_idx, new_idx):
        """Check if new_idx can be added to family_idx without creating a sunflower."""
        m = len(family_idx)
        nm = masks[new_idx]
        for i in range(m):
            fi = family_idx[i]
            a_new = inter[fi][new_idx]  # A ∩ S
            for j in range(i + 1, m):
                fj = family_idx[j]
                # Check if A ∩ B == A ∩ S == B ∩ S
                # A ∩ B = inter[fi][fj]
                # B ∩ S = inter[fj][new_idx]
                if inter[fi][fj] == a_new and inter[fj][new_idx] == a_new:
                    return False
        return True
    
    best_family = []
    best_size = 0
    
    random.seed(42)
    
    # Strategy: Multiple random greedy passes with local search refinement
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        
        # Random greedy initialization
        indices = list(range(num_sets))
        random.shuffle(indices)
        
        family_idx = []
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
            if can_add(family_idx, idx):
                family_idx.append(idx)
        
        current_size = len(family_idx)
        
        # Local search: try to add more sets
        fam_set = set(family_idx)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding random available sets
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                idx = random.randint(0, num_sets - 1)
                if idx in fam_set:
                    continue
                if can_add(family_idx, idx):
                    family_idx.append(idx)
                    fam_set.add(idx)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_family = family_idx[:]
        
        # If we have a good family, try swap-based local search
        if current_size > best_size * 0.9 and time.time() < deadline - 10:
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                # Pick a random set to remove
                if not family_idx:
                    break
                rem_pos = random.randint(0, len(family_idx) - 1)
                rem_idx = family_idx[rem_pos]
                family_idx.pop(rem_pos)
                fam_set.discard(rem_idx)
                current_size -= 1
                
                # Try to add a different set
                added = False
                for _ in range(100):
                    if time.time() > deadline - 10:
                        break
                    idx = random.randint(0, num_sets - 1)
                    if idx in fam_set:
                        continue
                    if can_add(family_idx, idx):
                        family_idx.append(idx)
                        fam_set.add(idx)
                        current_size += 1
                        added = True
                        break
                
                if not added:
                    # Restore
                    family_idx.append(rem_idx)
                    fam_set.add(rem_idx)
                    current_size += 1
                    continue
                
                if current_size > best_size:
                    best_size = current_size
                    best_family = family_idx[:]
        
        if best_size >= 250:
            break
    
    # Convert back to sorted lists
    return [sorted(all_subsets[i]) for i in best_family]
