import time
import random
import numpy as np
from itertools import combinations

# Precompute all 4-subsets of [1, 16]
ALL_SETS = [frozenset(c) for c in combinations(range(1, 17), 4)]
N_SETS = len(ALL_SETS)

# Convert to bitmasks for fast intersection
ALL_MASKS = [0] * N_SETS
for i, s in enumerate(ALL_SETS):
    m = 0
    for bit in s:
        m |= (1 << (bit - 1))
    ALL_MASKS[i] = m

# Precompute intersection matrix (N_SETS x N_SETS)
INTER = np.zeros((N_SETS, N_SETS), dtype=np.int32)
for i in range(N_SETS):
    mi = ALL_MASKS[i]
    for j in range(i, N_SETS):
        v = mi & ALL_MASKS[j]
        INTER[i, j] = v
        INTER[j, i] = v

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Helper to check if a set can be added to a family
    # family is a list of indices
    def can_add(idx_new, family):
        if len(family) < 2:
            return True
        
        # Convert family to numpy array for vectorized operations
        fam_arr = np.array(family, dtype=np.int32)
        
        # Get intersections of idx_new with all family members
        x_fam = INTER[idx_new, fam_arr]
        
        # Get pairwise intersections within family
        # This creates a |F| x |F| matrix
        # We only need upper triangle
        fam_pair = INTER[fam_arr[:, None], fam_arr[None, :]]
        
        # We need to check if there exist i < j such that:
        # x_fam[i] == x_fam[j] == fam_pair[i, j]
        
        # Vectorized approach:
        # For each pair (i, j), check if x_fam[i] == x_fam[j] and x_fam[i] == fam_pair[i, j]
        
        # Create a boolean matrix where M[i, j] is True if x_fam[i] == x_fam[j] == fam_pair[i, j]
        # M[i, j] = (x_fam[i] == x_fam[j]) & (x_fam[i] == fam_pair[i, j])
        
        # x_fam[i] == x_fam[j]:
        # We can broadcast x_fam[:, None] and x_fam[None, :]
        x_i = x_fam[:, None]
        x_j = x_fam[None, :]
        
        # x_fam[i] == fam_pair[i, j]
        # fam_pair[i, j] is already aligned
        
        # Condition: x_i == x_j AND x_i == fam_pair
        # But we only care about i < j
        
        cond = (x_i == x_j) & (x_i == fam_pair)
        
        # Mask out i >= j
        n_fam = len(family)
        mask = np.triu(np.ones((n_fam, n_fam), dtype=bool), k=1)
        cond = cond & mask
        
        return not np.any(cond)

    # Greedy construction
    def greedy(order):
        family = []
        for idx in order:
            if time.time() > deadline - 2.0:
                break
            if can_add(idx, family):
                family.append(idx)
        return family

    # Local search
    def local_search(family):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            in_family = set(family)
            not_in_family = [i for i in range(N_SETS) if i not in in_family]
            
            # 1. Try to add
            random.shuffle(not_in_family)
            for idx_new in not_in_family:
                if time.time() > deadline - 5.0:
                    return family
                if can_add(idx_new, family):
                    family.append(idx_new)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swap (1 for 1)
            if family:
                idx_remove = random.choice(family)
                not_in_family = [i for i in range(N_SETS) if i != idx_remove and i not in in_family]
                random.shuffle(not_in_family)
                
                for idx_add in not_in_family:
                    if time.time() > deadline - 5.0:
                        return family
                    temp_family = [x for x in family if x != idx_remove]
                    if can_add(idx_add, temp_family):
                        family = temp_family + [idx_add]
                        in_family = set(family)
                        improved = True
                        break
            
            if improved:
                continue

            # 3. Try 2-swap (remove 2, add 2)
            if len(family) >= 2:
                to_remove = random.sample(family, 2)
                in_remove = set(to_remove)
                temp_family = [x for x in family if x not in in_remove]
                not_in_family = [i for i in range(N_SETS) if i not in set(family)]
                random.shuffle(not_in_family)
                
                # Try to find 2 that can be added
                found = False
                for i in range(min(30, len(not_in_family))):
                    if time.time() > deadline - 5.0:
                        return family
                    idx_a = not_in_family[i]
                    if idx_a in in_remove:
                        continue
                    if not can_add(idx_a, temp_family):
                        continue
                    
                    # Now check if idx_b can be added to temp_family + [idx_a]
                    for j in range(i + 1, min(30, len(not_in_family))):
                        idx_b = not_in_family[j]
                        if idx_b in in_remove:
                            continue
                        if can_add(idx_b, temp_family + [idx_a]):
                            # Valid!
                            family = temp_family + [idx_a, idx_b]
                            found = True
                            improved = True
                            break
                    if found:
                        break
            
            if improved:
                continue

        return family

    # Main Loop
    random.seed(42)
    
    best_family = []
    best_size = 0
    
    # Phase 1: Multiple Greedy Restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        order = list(range(N_SETS))
        random.shuffle(order)
        
        family = greedy(order)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
    # Phase 2: Local Search on Best Family
    if best_size > 0 and time.time() < deadline - 5.0:
        current_family = best_family[:]
        
        # Run local search multiple times or until no improvement
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            
            improved_family = local_search(current_family)
            
            if len(improved_family) > len(current_family):
                current_family = improved_family
                if len(current_family) > best_size:
                    best_size = len(current_family)
                    best_family = current_family[:]
            else:
                # If no improvement, perturb and try again
                if time.time() > deadline - 2.0:
                    break
                if len(current_family) > 2:
                    to_remove = random.sample(current_family, 2)
                    current_family = [x for x in current_family if x not in to_remove]
                else:
                    break

    # Convert back to sorted lists
    return [sorted(list(ALL_SETS[idx])) for idx in best_family]
