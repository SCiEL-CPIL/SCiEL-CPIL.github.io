import random
import time
from itertools import combinations
from collections import defaultdict

def _is_valid_addition(C, family):
    """Check if adding C to family doesn't create a sunflower with two existing sets."""
    n = len(family)
    if n < 2:
        return True
    
    # Compute intersection of each family member with C
    inter_list = [A & C for A in family]
    
    # Group indices by intersection value
    groups = defaultdict(list)
    for i, inter in enumerate(inter_list):
        groups[inter].append(i)
    
    # For each group with >= 2 members, check if any pair has intersection == group key
    for inter, indices in groups.items():
        k = len(indices)
        if k < 2:
            continue
        for ii in range(k):
            i = indices[ii]
            A = family[i]
            for jj in range(ii + 1, k):
                j = indices[jj]
                B = family[j]
                if A & B == inter:
                    return False
    return True

def _greedy(order, family):
    """Greedily add sets from order if valid."""
    for C in order:
        if _is_valid_addition(C, family):
            family.append(C)

def construct():
    deadline = time.time() + 85.0
    
    # Generate all 4-element subsets of [1,16]
    all_sets = [set(s) for s in combinations(range(1, 17), 4)]
    
    best_family = []
    best_size = 0
    
    # Random greedy restarts
    for restart in range(2000):
        if time.time() > deadline - 3.0:
            break
        
        order = list(all_sets)
        random.shuffle(order)
        
        family = []
        _greedy(order, family)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
    
    # Local search on best family
    if time.time() < deadline - 2.0 and best_family:
        family = list(best_family)
        family_set = set(family)
        
        for attempt in range(500):
            if time.time() > deadline - 2.0:
                break
            
            idx = random.randint(0, len(family) - 1)
            removed = family[idx]
            temp = family[:idx] + family[idx + 1:]
            
            # Try a few random candidates
            candidates = random.sample(all_sets, min(100, len(all_sets)))
            found = False
            for C in candidates:
                if C in family_set:
                    continue
                if _is_valid_addition(C, temp):
                    family = temp + [C]
                    family_set = set(family)
                    found = True
                    break
            
            if not found:
                # Try to add without removing (if possible)
                for C in candidates:
                    if C in family_set:
                        continue
                    if _is_valid_addition(C, family):
                        family.append(C)
                        family_set = set(family)
                        found = True
                        break
        
        if len(family) > best_size:
            best_family = family
            best_size = len(family)
    
    return [sorted(s) for s in best_family]
