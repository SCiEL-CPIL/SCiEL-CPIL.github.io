import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = 75.0  # 75 seconds to be safe
    
    n = 16
    k = 4
    universe = frozenset(range(1, n + 1))
    
    all_sets = [set(c) for c in itertools.combinations(range(1, n + 1), k)]
    all_frozensets = [frozenset(s) for s in all_sets]
    
    best_family = []
    best_size = 0
    
    indices = list(range(len(all_sets)))
    
    # We will try multiple restarts
    # If we find a family of size > 1000, we might stop early?
    # The maximum is 1820.
    
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(indices)
        family = set()
        family_list = []
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
                
            S = all_frozensets[idx]
            
            conflict = False
            for A in family_list:
                K = A & S
                len_K = len(K)
                needed_size = 4 - len_K
                
                # Quick bounds
                # |A U S| = |A| + |S| - |K| = 8 - len_K
                # |U| = 16 - (8 - len_K) = 8 + len_K
                # We need to pick needed_size elements from U.
                # If needed_size > 8 + len_K, impossible. (Always true here as needed_size <= 4, 8+len_K >= 8)
                
                U = universe - (A | S)
                
                if needed_size == 0:
                    if K in family and K != A:
                        conflict = True
                        break
                elif needed_size == 1:
                    # Check if any K | {x} in family
                    for x in U:
                        B = K | {x}
                        if B in family and B != A:
                            conflict = True
                            break
                    if conflict: break
                elif needed_size == 2:
                    for x, y in itertools.combinations(U, 2):
                        B = K | {x, y}
                        if B in family and B != A:
                            conflict = True
                            break
                    if conflict: break
                elif needed_size == 3:
                    for x, y, z in itertools.combinations(U, 3):
                        B = K | {x, y, z}
                        if B in family and B != A:
                            conflict = True
                            break
                    if conflict: break
                elif needed_size == 4:
                    for D in itertools.combinations(U, 4):
                        B = frozenset(D)
                        if B in family and B != A:
                            conflict = True
                            break
                    if conflict: break
                    
            if not conflict:
                family.add(S)
                family_list.append(S)
                
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
            
    return [sorted(list(s)) for s in best_family]
