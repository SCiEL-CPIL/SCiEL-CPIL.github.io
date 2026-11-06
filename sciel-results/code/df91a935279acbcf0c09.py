"""Rewritten program for sunflower-free 4-uniform family on [1,16]."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def _precompute_pair_intersections(all_sets):
    """Precompute pairwise intersections for all sets."""
    n = len(all_sets)
    pair_int = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            v = all_sets[i] & all_sets[j]
            pair_int[i][j] = v
            pair_int[j][i] = v
    return pair_int

def _check_sunflower_fast(new_idx, family_indices, all_sets, pair_int):
    """
    Check if adding new_idx to family creates a sunflower.
    Uses grouping by intersection with new set.
    """
    s = all_sets[new_idx]
    # Group family members by their intersection with s
    groups = defaultdict(list)
    for fi in family_indices:
        inter = all_sets[fi] & s
        groups[inter].append(fi)
    
    # For each group with >= 2 members, check if any pair has intersection == group key
    for inter, members in groups.items():
        if len(members) < 2:
            continue
        # Check all pairs in this group
        for a_pos in range(len(members)):
            for b_pos in range(a_pos+1, len(members)):
                a = members[a_pos]
                b = members[b_pos]
                if pair_int[a][b] == inter:
                    return True
    return False

def _greedy_construct(all_sets, pair_int, order, deadline):
    """Run greedy construction with given order."""
    family = []
    family_set = set()
    
    for idx in order:
        if time.time() > deadline:
            break
        if idx in family_set:
            continue
        if not _check_sunflower_fast(idx, family, all_sets, pair_int):
            family.append(idx)
            family_set.add(idx)
    
    return family

def _local_search_swap(family, all_sets, pair_int, deadline):
    """
    Local search: try to improve family by swap moves (remove one, add one)
    or by adding if possible.
    """
    family_set = set(family)
    n_total = len(all_sets)
    improved = True
    max_iterations = 200
    
    while improved and time.time() < deadline and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        # First, try adding any set that doesn't conflict
        for idx in range(n_total):
            if time.time() > deadline:
                break
            if idx in family_set:
                continue
            if not _check_sunflower_fast(idx, list(family_set), all_sets, pair_int):
                family.append(idx)
                family_set.add(idx)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove one, add one
        # Try removing each family member and see if we can add a better set
        random.shuffle(family)
        for remove_idx in family:
            if time.time() > deadline:
                break
            
            temp_family = [x for x in family if x != remove_idx]
            temp_set = set(temp_family)
            
            # Find best candidate to add
            best_add = -1
            for add_idx in range(n_total):
                if add_idx in temp_set:
                    continue
                if not _check_sunflower_fast(add_idx, temp_family, all_sets, pair_int):
                    best_add = add_idx
                    break
            
            if best_add != -1:
                family = temp_family
                family.append(best_add)
                family_set = set(family)
                improved = True
                break
    
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill
    
    # Represent sets as bitmasks for speed
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    n_total = len(all_sets)  # C(16,4) = 1820
    
    # Precompute pair intersections
    pair_int = _precompute_pair_intersections(all_sets)
    
    best_family = []
    best_size = 0
    
    # Multiple greedy passes with different orderings
    n_attempts = 20
    for attempt in range(n_attempts):
        if time.time() > deadline - 15:
            break
        
        if attempt == 0:
            order = list(range(n_total))
        elif attempt == 1:
            order = list(range(n_total - 1, -1, -1))
        else:
            order = list(range(n_total))
            random.shuffle(order)
        
        family = _greedy_construct(all_sets, pair_int, order, deadline - 15)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        
        if best_size >= 250:
            break
    
    # Local search on the best family
    if best_size > 0 and time.time() < deadline - 5:
        improved_family = _local_search_swap(best_family, all_sets, pair_int, deadline - 3)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family = improved_family
    
    # Convert back to sorted lists
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END
