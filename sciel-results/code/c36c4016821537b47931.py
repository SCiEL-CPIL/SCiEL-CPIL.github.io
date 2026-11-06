"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # All 4-subsets of [1,16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    n = len(all_sets)
    
    # Precompute pairwise intersections
    # For each pair (i, j), compute the intersection size and the intersection set
    # A sunflower of 3 sets A, B, C: all three pairwise intersections are equal
    # i.e., A∩B = A∩C = B∩C
    
    # Key insight: For three 4-sets, a sunflower means there exists a set S such that
    # A∩B = S, A∩C = S, B∩C = S
    # This means A, B, C all contain S, and outside S, they are pairwise disjoint.
    # Since |A|=|B|=|C|=4, if |S|=s, then each set has 4-s elements outside S,
    # and these outside parts must be pairwise disjoint.
    
    # So for a given core S of size s (s=0,1,2,3), we need three sets that all contain S
    # and whose complements (relative to S) are pairwise disjoint.
    
    # Let's precompute for each pair of sets, their intersection
    # Then for a candidate set C, we need to check if there exist A, B already chosen
    # such that A∩B = A∩C = B∩C
    
    # More efficient: for a new set C, and existing sets A, B:
    # A∩C = B∩C = A∩B
    # Let X = A∩C. Then B∩C = X and A∩B = X.
    # This means: A contains X, B contains X, C contains X.
    # A\X and B\X are disjoint from C\X and from each other... 
    # Actually: A∩B = X means A\X and B\X are disjoint.
    # A∩C = X means A\X and C\X are disjoint.
    # B∩C = X means B\X and C\X are disjoint.
    # So A\X, B\X, C\X are pairwise disjoint, and each has size 4-|X|.
    
    # For checking if adding C creates a sunflower with A, B:
    # We need A∩B = A∩C = B∩C
    # Let's compute this directly.
    
    # Strategy: Greedy with randomization and local search
    # 1. Build initial solution greedily
    # 2. Try to add more sets
    # 3. Use simulated annealing or local search to improve
    
    # For efficiency, precompute for each set its elements as a bitmask
    # Use bitmasks for fast intersection
    
    # Represent each set as a 16-bit integer
    set_to_mask = {}
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x-1))
        set_to_mask[frozenset(s)] = mask
    
    # Actually, let's work with indices and precomputed masks
    masks = []
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x-1))
        masks.append(mask)
    
    # Precompute intersection masks for all pairs? That's 1820*1819/2 ≈ 1.65M pairs
    # Too many to precompute all. Let's compute on the fly but use bit operations.
    
    def is_sunflower_triple(i, j, k, masks):
        """Check if sets i, j, k form a sunflower (all three pairwise intersections equal)."""
        ij = masks[i] & masks[j]
        ik = masks[i] & masks[k]
        jk = masks[j] & masks[k]
        return ij == ik == jk
    
    def can_add(candidate_idx, chosen_indices, masks):
        """Check if adding candidate_idx to chosen_indices creates any sunflower triple."""
        mask_c = masks[candidate_idx]
        for ai in range(len(chosen_indices)):
            i = chosen_indices[ai]
            ic = masks[i] & mask_c
            for bi in range(ai+1, len(chosen_indices)):
                j = chosen_indices[bi]
                ij = masks[i] & masks[j]
                jc = masks[j] & mask_c
                if ij == ic == jc:
                    return False
        return True
    
    # Greedy construction with randomization
    best_solution = []
    
    def greedy_build(order):
        """Build a solution by processing sets in the given order."""
        chosen = []
        for idx in order:
            if can_add(idx, chosen, masks):
                chosen.append(idx)
        return chosen
    
    # Try multiple random orderings
    best_size = 0
    best_chosen = []
    
    # First, try a deterministic order (sorted by some heuristic)
    # Heuristic: prefer sets with smaller "conflict potential"
    # For now, just try random orders
    
    num_trials = 50
    trial = 0
    
    while trial < num_trials and time.time() < deadline:
        order = list(range(n))
        random.shuffle(order)
        
        # Try greedy
        chosen = greedy_build(order)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = chosen[:]
        
        trial += 1
    
    # Local search: try to add more sets
    if time.time() < deadline:
        chosen_set = set(best_chosen)
        remaining = [i for i in range(n) if i not in chosen_set]
        
        # Try adding sets from remaining
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            remaining = [i for i in remaining if i not in chosen_set]
            for idx in remaining[:]:
                if can_add(idx, list(best_chosen), masks):
                    best_chosen.append(idx)
                    chosen_set.add(idx)
                    added_any = True
                    break
    
    # Convert back to sorted lists
    result = [sorted(all_sets[i]) for i in best_chosen]
    
    return result

# EVOLVE-BLOCK-END
