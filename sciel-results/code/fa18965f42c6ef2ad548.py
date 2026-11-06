import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks (indices 0..15)
    all_masks = []
    for c in combinations(range(n), k):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    total_sets = len(all_masks)  # 1820
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}
    
    # Precompute pairwise intersections for all pairs of masks
    # inter[i][j] = all_masks[i] & all_masks[j]
    # This is a 1820x1820 matrix, which is ~3.3M entries, fine for memory
    # Actually, let's be smarter. We only need to check if adding C to family F
    # creates a sunflower. A sunflower is three sets A, B, C such that
    # A&B = A&C = B&C.
    # So for a candidate C, we need to check all pairs (A, B) in F such that
    # (A & B) == (A & C) == (B & C).
    
    # Let's precompute for each pair of masks (i, j), the intersection value.
    # But that's 1820*1820/2 = ~1.6M values. That's fine.
    # Actually, we can precompute a dictionary: for each possible intersection value V,
    # which pairs (i, j) have intersection V? No, that's not directly helpful.
    
    # Better approach: For each candidate C, and each pair (A, B) in the current family,
    # check if A&B == A&C == B&C.
    # We can precompute A&C for all A and C? That's 1820*1820 = 3.3M operations, which is fine.
    # Let's precompute a 2D array inter[i][j] = all_masks[i] & all_masks[j]
    
    inter = [[0] * total_sets for _ in range(total_sets)]
    for i in range(total_sets):
        mi = all_masks[i]
        for j in range(i + 1, total_sets):
            ij = mi & all_masks[j]
            inter[i][j] = ij
            inter[j][i] = ij
    
    def can_add(C_idx, family_indices):
        """Check if mask C_idx can be added to family (list of indices)."""
        C = all_masks[C_idx]
        fam = family_indices
        n_fam = len(fam)
        if n_fam < 2:
            return True
        
        # For each pair (A, B) in family, check if A&B == A&C == B&C
        # We can precompute A&C for all A in family
        # Let's just iterate
        for a_idx in range(n_fam):
            A = all_masks[fam[a_idx]]
            AC = A & C
            for b_idx in range(a_idx + 1, n_fam):
                B = all_masks[fam[b_idx]]
                AB = inter[fam[a_idx]][fam[b_idx]]
                # Check if AC == AB and (B & C) == AB
                if AC == AB:
                    BC = B & C
                    if BC == AB:
                        return False
        return True
    
    best_family_size = 0
    best_family = []
    
    # Strategy: Multiple random restarts with greedy construction, then local search
    # to try to add more sets.
    
    while time.time() < deadline:
        random.shuffle(all_masks)
        family = []
        family_set = set()
        
        for mask in all_masks:
            idx = mask_to_idx[mask]
            if can_add(idx, family):
                family.append(idx)
                family_set.add(idx)
        
        # Local search: try to add any remaining set
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for mask in all_masks:
                idx = mask_to_idx[mask]
                if idx not in family_set:
                    if can_add(idx, family):
                        family.append(idx)
                        family_set.add(idx)
                        improved = True
        
        # Try swaps: remove one, add one, if it allows adding more later?
        # Actually, simple swaps might not help if we're stuck.
        # Let's try: for each pair (remove, add), if adding 'add' to (family - remove) is valid,
        # and the new family is at least as large, keep it.
        # But we want to maximize size, so we only accept if we can then add more.
        # This is complex. Let's just do simple greedy + fill.
        
        if len(family) > best_family_size:
            best_family_size = len(family)
            best_family = family[:]
            
            # If we get a very good size, we might break early?
            # Max possible is 250. If we reach 200+, it's good.
            if best_family_size >= 200:
                # Keep trying to improve, but we have a good candidate
                pass

    # Convert bitmasks back to sorted lists of integers (1-indexed)
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
