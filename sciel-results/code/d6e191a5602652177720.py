import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute all 4-subsets of [1,16]
    all_sets = [set(s) for s in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Precompute frozenset representations
    all_frozensets = [frozenset(s) for s in all_sets]
    
    # For each set, precompute its intersection with every other set
    # intersections[i][j] = frozenset(all_sets[i] & all_sets[j])
    # This is 1820x1820 = ~3.3M entries, each a frozenset of size 0-4
    # Memory is fine, but let's be smart: only compute once
    
    # Actually, let's use a different approach for speed:
    # For a candidate to be added to family F, we need:
    # For no two existing sets A, B in F: (cand & A) == (cand & B) == (A & B)
    # 
    # Equivalently: for each intersection value S = cand & A (for A in F),
    #   the number of B in F such that (cand & B) == S AND (A & B) == S must be 0
    #   for all A with cand & A == S.
    #
    # Simpler: group F by (cand & x). For group G with intersection S,
    #   check if any pair in G has intersection exactly S.
    
    def is_valid_addition_fast(cand, cand_fs, family, family_frozensets):
        """Check if cand can be added to family without creating a sunflower."""
        # Group family members by their intersection with cand
        groups = {}
        for i, existing in enumerate(family):
            inter = cand_fs & family_frozensets[i]
            if inter not in groups:
                groups[inter] = [i]
            else:
                groups[inter].append(i)
        
        # For each group with intersection S, check if any pair has intersection S
        for S, indices in groups.items():
            if len(indices) < 2:
                continue
            # Check all pairs in this group
            n = len(indices)
            for a in range(n):
                ia = indices[a]
                for b in range(a+1, n):
                    ib = indices[b]
                    if family_frozensets[ia] & family_frozensets[ib] == S:
                        return False
        return True
    
    best_family = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Strategy: Multiple greedy passes with different orderings, then local search
    # Key speedup: use frozensets for fast intersection
    
    num_orders = 20
    for trial in range(num_orders):
        if time.time() > deadline - 5:
            break
        
        if trial == 0:
            order = list(range(n_sets))
        else:
            order = list(range(n_sets))
            rng.shuffle(order)
        
        family = []
        family_frozensets = []
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            
            cand = all_sets[idx]
            cand_fs = all_frozensets[idx]
            if is_valid_addition_fast(cand, cand_fs, family, family_frozensets):
                family.append(cand)
                family_frozensets.append(cand_fs)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
            best_family_fs = list(family_frozensets)
    
    # Local search: try to add more sets
    if time.time() < deadline - 3 and best_family:
        family = list(best_family)
        family_frozensets = list(best_family_fs)
        family_set = set(family_frozensets)
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try random additions
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                idx = rng.randint(0, n_sets - 1)
                cand_fs = all_frozensets[idx]
                if cand_fs not in family_set:
                    cand = all_sets[idx]
                    if is_valid_addition_fast(cand, cand_fs, family, family_frozensets):
                        family.append(cand)
                        family_frozensets.append(cand_fs)
                        family_set.add(cand_fs)
                        improved = True
                        best_size = len(family)
                        best_family = list(family)
                        best_family_fs = list(family_frozensets)
    
    # Second round of local search with more iterations if time permits
    if time.time() < deadline - 2 and best_family:
        family = list(best_family)
        family_frozensets = list(best_family_fs)
        family_set = set(family_frozensets)
        
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            for _ in range(100):
                idx = rng.randint(0, n_sets - 1)
                cand_fs = all_frozensets[idx]
                if cand_fs not in family_set:
                    cand = all_sets[idx]
                    if is_valid_addition_fast(cand, cand_fs, family, family_frozensets):
                        family.append(cand)
                        family_frozensets.append(cand_fs)
                        family_set.add(cand_fs)
                        if len(family) > best_size:
                            best_size = len(family)
                            best_family = list(family)
                            best_family_fs = list(family_frozensets)
    
    return [sorted(s) for s in best_family]
