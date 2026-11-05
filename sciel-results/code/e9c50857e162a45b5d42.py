import random
from itertools import combinations
from math import lcm, gcd

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute forbidden triples map: pair -> set of conflicting third elements
    forbidden_by_pair = {}
    
    for a, b, c in combinations(numbers, 3):
        if lcm(a, b) == lcm(a, c) == lcm(b, c):
            forbidden_by_pair.setdefault(tuple(sorted((a, b))), set()).add(c)
            forbidden_by_pair.setdefault(tuple(sorted((a, c))), set()).add(b)
            forbidden_by_pair.setdefault(tuple(sorted((b, c))), set()).add(a)
            
    empty_set = frozenset()
    
    def is_valid_addition(x, current_set):
        for a, b in combinations(current_set, 2):
            key = tuple(sorted((a, b)))
            if x in forbidden_by_pair.get(key, empty_set):
                return False
        return True

    # 1. Initialize with a Greedy Set
    S = set()
    # Randomize initial order to avoid deterministic bias? 
    # Or use 1..128. Let's use 1..128 for reproducibility in init, 
    # but local search will handle the rest.
    for x in numbers:
        if is_valid_addition(x, S):
            S.add(x)
            
    T = set(numbers) - S
    
    improved = True
    max_iterations = 100 # Each iteration is fast now
    
    while improved and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        # Phase 1: Try to add any element from T to S
        added = False
        # Shuffle T to vary the order of additions
        T_list = list(T)
        random.shuffle(T_list)
        for y in T_list:
            if is_valid_addition(y, S):
                S.add(y)
                T.remove(y)
                added = True
                improved = True
                break # Restart loop to re-scan T
                
        if added:
            continue
            
        # Phase 2: Try swaps
        # For each y in T, check if there is an x in S such that 
        # removing x makes y valid.
        # This happens if all pairs in S that block y contain x.
        
        T_list = list(T)
        random.shuffle(T_list)
        
        found_swap = False
        for y in T_list:
            if found_swap:
                break
                
            # Find all pairs in S that block y
            blocking_pairs = []
            S_list = list(S)
            # Optimization: If S is large, this is O(|S|^2)
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i + 1, len(S_list)):
                    b = S_list[j]
                    key = tuple(sorted((a, b)))
                    if y in forbidden_by_pair.get(key, empty_set):
                        blocking_pairs.append((a, b))
                        
            if not blocking_pairs:
                # Should have been caught in Add phase, but just in case
                # (e.g. if T_list was stale? No, we rebuilt it)
                # If no blocking pairs, y is valid to add.
                S.add(y)
                T.remove(y)
                improved = True
                found_swap = True # Actually it's an add, but logic flow works
                break
                
            # Find common elements in all blocking pairs
            # If a pair (a,b) blocks y, x must be a or b to remove that block.
            # So x must be in the intersection of all blocking pairs' elements.
            common = set(blocking_pairs[0])
            for pair in blocking_pairs[1:]:
                common.intersection_update(pair)
                
            if common:
                # Pick an x from common
                x = next(iter(common))
                # Perform swap
                S.remove(x)
                T.remove(y)
                S.add(y)
                T.add(x)
                improved = True
                found_swap = True
                break
                
    if not S:
        return [1]
        
    return list(S)
