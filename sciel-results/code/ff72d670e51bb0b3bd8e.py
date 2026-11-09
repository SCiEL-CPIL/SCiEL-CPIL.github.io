import time
from math import gcd
from random import Random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    # Precompute gcd table
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    def is_valid_set(s):
        """Check if a set is valid."""
        lst = list(s)
        m = len(lst)
        for i in range(m):
            a = lst[i]
            for j in range(i+1, m):
                b = lst[j]
                gab = gcd_table[a][b]
                for k in range(j+1, m):
                    c = lst[k]
                    if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                        return False
        return True

    def valid_addition(selected_list, x):
        """Check if adding x to selected_list maintains validity."""
        m = len(selected_list)
        for i in range(m):
            a = selected_list[i]
            for j in range(i+1, m):
                b = selected_list[j]
                gab = gcd_table[a][b]
                if gcd_table[a][x] == gab and gcd_table[b][x] == gab:
                    return False
        return True

    def local_search(selected_set, rng):
        selected_list = list(selected_set)
        best_list = selected_list[:]
        best_size = len(best_list)
        remaining = [x for x in range(1, n+1) if x not in selected_set]
        
        # Try adding
        for x in remaining:
            if valid_addition(best_list, x):
                best_list.append(x)
                best_size += 1
        
        # Try swaps
        improved = True
        while improved:
            improved = False
            for i in range(len(best_list)):
                orig = best_list[i]
                trial = best_list[:i] + best_list[i+1:]
                for x in remaining:
                    if x in trial: continue
                    if valid_addition(trial, x):
                        new_list = trial + [x]
                        if len(new_list) > best_size:
                            best_list = new_list
                            best_size = len(new_list)
                            remaining.remove(orig)
                            remaining.append(x)
                            improved = True
                            break
                if improved: break
        
        return best_list

    best_set = set()
    
    # Strategy 1: Greedy from random start
    for restart in range(500):
        if time.time() > deadline - 5:
            break
        
        rng = Random(restart * 1000003 + 7)
        candidates = list(range(1, n+1))
        rng.shuffle(candidates)
        
        current = set()
        for x in candidates:
            if valid_addition(list(current), x):
                current.add(x)
        
        improved = local_search(current, rng)
        
        if len(improved) > len(best_set):
            best_set = set(improved)
            if time.time() > deadline - 10:
                break

    # Strategy 2: Start with specific structured sets
    # Powers of 2
    seed1 = set([1, 2, 4, 8, 16, 32, 64, 128])
    improved = local_search(seed1, Random(123))
    if len(improved) > len(best_set):
        best_set = set(improved)
        
    # All even
    seed2 = set(range(2, n+1, 2))
    # This is likely invalid, but local search might fix it?
    # Actually, local search only adds/swaps, doesn't remove to fix violations if start is invalid.
    # So we must start with valid sets.
    
    # All multiples of 4
    seed3 = set(range(4, n+1, 4))
    # Check if valid
    if is_valid_set(seed3):
        improved = local_search(seed3, Random(456))
        if len(improved) > len(best_set):
            best_set = set(improved)
            
    # All numbers coprime to 6?
    seed4 = set(x for x in range(1, n+1) if gcd(x, 6) == 1)
    if is_valid_set(seed4):
        improved = local_search(seed4, Random(789))
        if len(improved) > len(best_set):
            best_set = set(improved)
            
    # Final result
    if not best_set:
        return [1]
        
    return sorted(best_set)
