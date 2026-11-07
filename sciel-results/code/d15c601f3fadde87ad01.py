import itertools
import random
import math

def construct():
    n = 12
    m = 40
    
    # 1. Generate all subsets and map to indices
    all_subsets = []
    for r in range(n+1):
        for combo in itertools.combinations(range(n), r):
            all_subsets.append(frozenset(combo))
            
    num_subsets = len(all_subsets) # 4096
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # 2. Precompute comparability masks
    # comp_masks[i] is an integer with bit j set if all_subsets[i] is comparable to all_subsets[j]
    comp_masks = [0] * num_subsets
    for i in range(num_subsets):
        si = all_subsets[i]
        mask = 0
        for j in range(num_subsets):
            sj = all_subsets[j]
            if si < sj or sj < si:
                mask |= (1 << j)
        comp_masks[i] = mask
        
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
        
    # 3. Heuristic Initialization: "Small Sets"
    # We want to pick sets that are likely to be comparable.
    # Small sets have many supersets.
    # Let's pick 1 set of size 0, 12 sets of size 1, and distribute the rest
    # between size 2 and 3 based on a simple heuristic or just pick specific ones.
    
    # Indices for size 0
    size_0_indices = [subset_to_idx[frozenset()]]
    # Indices for size 1
    size_1_indices = [subset_to_idx[frozenset([i])] for i in range(n)]
    
    # We need m - 13 = 27 more sets.
    # Let's pick 10 from size 2 and 17 from size 3.
    # To maximize comparable pairs between size 2 and 3, we should pick size 3 sets
    # that contain the chosen size 2 sets.
    
    # Pick 10 size 2 sets. Let's pick the first 10 combinations.
    size_2_candidates = []
    for combo in itertools.combinations(range(n), 2):
        size_2_candidates.append(subset_to_idx[frozenset(combo)])
        if len(size_2_candidates) == 10:
            break
            
    # Pick 17 size 3 sets.
    # Ideally, they should contain the size 2 sets.
    # But any size 3 set contains 3 size 2 sets.
    # Let's just pick the first 17 size 3 sets for now, and let SA optimize.
    size_3_candidates = []
    for combo in itertools.combinations(range(n), 3):
        size_3_candidates.append(subset_to_idx[frozenset(combo)])
        if len(size_3_candidates) == 17:
            break
            
    init_indices = size_0_indices + size_1_indices + size_2_candidates + size_3_candidates
    
    # 4. Simulated Annealing
    
    def get_score_from_indices(indices):
        mask = 0
        for i in indices:
            mask |= (1 << i)
        score = 0
        for i in indices:
            score += popcount(comp_masks[i] & mask)
        return score // 2
        
    current_family = list(init_indices)
    current_mask = 0
    for i in current_family:
        current_mask |= (1 << i)
    current_score = get_score_from_indices(current_family)
    
    best_score = current_score
    best_family = list(current_family)
    
    # SA Parameters
    T = 100.0
    T_min = 0.01
    cooling = 0.999
    max_steps = 5000
    
    for step in range(max_steps):
        if T < T_min:
            break
            
        # Pick two random positions to swap
        i = random.randint(0, m - 1)
        j = random.randint(0, m - 1)
        if i == j:
            continue
            
        old_i = current_family[i]
        old_j = current_family[j]
        
        # Calculate delta score
        # Remove old_i and old_j from mask
        other_mask = current_mask ^ (1 << old_i) ^ (1 << old_j)
        
        old_comps_i = popcount(comp_masks[old_i] & other_mask)
        old_comps_j = popcount(comp_masks[old_j] & other_mask)
        old_pair_comp = 1 if (comp_masks[old_i] & (1 << old_j)) else 0
        old_local_score = old_comps_i + old_comps_j + old_pair_comp
        
        # Pick new indices
        # Randomly pick two new indices not in current_family (excluding i,j)
        # To avoid bias, we can just pick random indices and check if they are in family.
        # But we must ensure they are distinct and not in family.
        
        new_i = -1
        new_j = -1
        
        # Try to find valid new_i
        while True:
            cand_i = random.randint(0, num_subsets - 1)
            if not (current_mask & (1 << cand_i)):
                new_i = cand_i
                break
                
        # Try to find valid new_j (must be different from new_i)
        while True:
            cand_j = random.randint(0, num_subsets - 1)
            if cand_j != new_i and not (current_mask & (1 << cand_j)):
                new_j = cand_j
                break
        
        # Calculate new local score
        new_comps_i = popcount(comp_masks[new_i] & other_mask)
        new_comps_j = popcount(comp_masks[new_j] & other_mask)
        new_pair_comp = 1 if (comp_masks[new_i] & (1 << new_j)) else 0
        new_local_score = new_comps_i + new_comps_j + new_pair_comp
        
        delta = new_local_score - old_local_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta > 0 or random.random() < math.exp(-delta / T):
            # Perform swap
            current_family[i] = new_i
            current_family[j] = new_j
            current_mask ^= (1 << old_i)
            current_mask ^= (1 << old_j)
            current_mask |= (1 << new_i)
            current_mask |= (1 << new_j)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family = list(current_family)
                
        T *= cooling
        
    # Convert back to lists of ints
    result = [sorted(list(all_subsets[idx])) for idx in best_family]
    return result
