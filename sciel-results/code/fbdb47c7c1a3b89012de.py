import itertools
import random

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
        
    # 3. Greedy Initialization
    family_indices = []
    family_set = set()
    
    # Start with empty set and full set? Or just greedy from scratch.
    # Let's use the greedy logic from before but with indices.
    # To make it fast, we can just pick random ones or use the previous greedy.
    # Previous greedy: 
    # family = [frozenset(), frozenset(range(n))]
    # Let's stick to that for consistency, then optimize.
    
    init_sets = [frozenset(), frozenset(range(n))]
    family_indices = [subset_to_idx[s] for s in init_sets]
    family_set = set(family_indices)
    
    remaining_indices = set(range(num_subsets)) - family_set
    
    while len(family_indices) < m:
        best_idx = -1
        best_gain = -1
        for idx in remaining_indices:
            s = all_subsets[idx]
            # Gain is number of sets in family comparable to s
            # We can use the mask!
            # But we need to intersect with family_mask.
            # Let's build family_mask as we go? Or just compute simply for init.
            # Init is only 40 steps, simple loop is fine.
            gain = sum(1 for i in family_indices if all_subsets[idx] < all_subsets[i] or all_subsets[i] < all_subsets[idx])
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        family_indices.append(best_idx)
        family_set.add(best_idx)
        remaining_indices.discard(best_idx)
        
    # 4. Local Search with Bitsets
    # We will run multiple restarts
    
    def get_score_from_indices(indices):
        mask = 0
        for i in indices:
            mask |= (1 << i)
        # Score is sum of popcount(comp_masks[i] & mask) for i in indices, divided by 2
        # Because each pair is counted twice.
        # Or just count pairs directly?
        # Using masks:
        # Total comparable pairs = 1/2 * sum( popcount(comp_masks[i] & mask) for i in indices )
        # Note: comp_masks[i] does NOT include i (since a set is not a strict subset of itself).
        # So we don't need to worry about self-loops.
        score = 0
        for i in indices:
            score += popcount(comp_masks[i] & mask)
        return score // 2
        
    best_score = get_score_from_indices(family_indices)
    best_family = list(family_indices)
    
    # Precompute family_mask for current family
    current_family = list(family_indices)
    current_mask = 0
    for i in current_family:
        current_mask |= (1 << i)
    current_score = best_score
        
    for restart in range(10): # Try 10 restarts
        if restart > 0:
            # Perturb
            current_family = list(best_family)
            current_mask = 0
            for i in current_family:
                current_mask |= (1 << i)
            current_score = best_score
            
            # Randomly swap 5 elements
            for _ in range(5):
                idx_pos = random.randint(0, m-1)
                old_idx = current_family[idx_pos]
                # Pick a random new index not in current_family
                # This is O(N) to find if we don't have a set.
                # We have current_mask. We can pick a random number and check bit?
                # Or just maintain a set of available indices.
                # For perturbation, let's just pick a random index from all_subsets and if it's in family, retry.
                while True:
                    new_idx = random.randint(0, num_subsets - 1)
                    if not (current_mask & (1 << new_idx)):
                        break
                
                current_family[idx_pos] = new_idx
                current_mask ^= (1 << old_idx)
                current_mask |= (1 << new_idx)
            
            # Recalculate score? Or update incrementally?
            # Recalculating is fast enough with bitsets (40 popcounts).
            current_score = get_score_from_indices(current_family)
            
        improved = True
        while improved:
            improved = False
            # Try to improve each position
            for i in range(m):
                old_idx = current_family[i]
                # Mask of other family members
                other_mask = current_mask ^ (1 << old_idx)
                
                old_comps = popcount(comp_masks[old_idx] & other_mask)
                
                best_new_idx = -1
                best_new_comps = old_comps
                
                # Try all candidates
                # We can iterate through all indices not in current_mask
                # To do this efficiently, we can iterate through all 4096 and skip those in mask?
                # Or generate a list of available indices?
                # List of 4056 indices.
                # Let's just iterate through all 4096 and check mask.
                # 4096 checks * 40 positions = 163,840 checks per pass.
                # Each check: 1 bit op + 1 popcount (if candidate).
                # Actually, we only need popcount if it's a candidate.
                
                for new_idx in range(num_subsets):
                    if current_mask & (1 << new_idx):
                        continue
                    new_comps = popcount(comp_masks[new_idx] & other_mask)
                    if new_comps > best_new_comps:
                        best_new_comps = new_comps
                        best_new_idx = new_idx
                        
                if best_new_idx != -1 and best_new_comps > old_comps:
                    # Perform swap
                    current_family[i] = best_new_idx
                    current_mask ^= (1 << old_idx)
                    current_mask |= (1 << best_new_idx)
                    current_score += best_new_comps - old_comps
                    improved = True
                    
        if current_score > best_score:
            best_score = current_score
            best_family = list(current_family)
            
    # Convert back to lists of ints
    result = [sorted(list(all_subsets[idx])) for idx in best_family]
    return result
