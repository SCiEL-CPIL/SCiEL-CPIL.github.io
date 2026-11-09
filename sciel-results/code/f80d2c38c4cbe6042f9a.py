"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    
    # Precompute gcd table for [1,128]
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute conflict pairs: for each x, which pairs (a,b) with a<b create a bad triple with x
    # conflict_pairs[x] = set of (a,b) such that {a,b,x} forms a bad triple
    conflict_pairs = [set() for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            g_ab = gcd_table[a][b]
            for c in range(1, n+1):
                if c == a or c == b:
                    continue
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    # c conflicts with pair (a,b)
                    conflict_pairs[c].add((a, b))
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
    
    def can_add(selected_set, x):
        """Check if adding x to selected_set maintains the no-equal-gcd-triple property."""
        # For each pair in selected_set, check if (a,b) is in conflict_pairs[x]
        sel = list(selected_set)
        m = len(sel)
        for i in range(m):
            a = sel[i]
            for j in range(i+1, m):
                b = sel[j]
                if (a, b) in conflict_pairs[x] or (b, a) in conflict_pairs[x]:
                    return False
        return True
    
    def local_search(selected_set, rng):
        """Try to improve the set by additions and swaps."""
        best_list = list(selected_set)
        best_size = len(best_list)
        
        # Repeatedly try to add elements until no more can be added
        while True:
            best_set = set(best_list)
            remaining = [x for x in range(1, n+1) if x not in best_set]
            rng.shuffle(remaining)
            added = False
            for x in remaining:
                if can_add(best_set, x):
                    best_list.append(x)
                    best_size = len(best_list)
                    added = True
                    break
            if not added:
                break
        
        # Try swaps: remove one element, then try to add multiple others
        improved = True
        while improved:
            improved = False
            best_list_set = set(best_list)
            remaining = [x for x in range(1, n+1) if x not in best_list_set]
            rng.shuffle(remaining)
            
            for u in list(best_list_set):
                if time.time() > deadline - 3:
                    return best_list
                
                trial = [x for x in best_list if x != u]
                
                # Try to add elements from remaining
                new_list = trial[:]
                new_size = len(new_list)
                
                for x in remaining:
                    if can_add(set(new_list), x):
                        new_list.append(x)
                        new_size = len(new_list)
                
                if new_size > best_size:
                    best_list = new_list
                    best_size = new_size
                    improved = True
                    break
        
        return best_list
    
    best_set = set()
    
    # Multiple random restarts with greedy construction
    for restart in range(2000):
        if time.time() > deadline - 5:
            break
        
        rng = Random(restart * 7919 + 13)
        
        # Greedy construction with random ordering
        candidates = list(range(1, n+1))
        rng.shuffle(candidates)
        
        current = set()
        for x in candidates:
            if can_add(current, x):
                current.add(x)
        
        # Local search to improve
        improved = local_search(current, rng)
        
        if len(improved) > len(best_set):
            best_set = set(improved)
            if time.time() > deadline - 10:
                break
    
    # Final local search on best set if time permits
    if time.time() < deadline - 5:
        rng = Random(42)
        final = local_search(best_set, rng)
        if len(final) > len(best_set):
            best_set = set(final)
    
    if not best_set:
        return [1]
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
