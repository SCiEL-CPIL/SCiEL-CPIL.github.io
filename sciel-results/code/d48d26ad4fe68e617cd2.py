import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 80.0  # Safety margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute pair conflicts
    # pair_conflicts[a][b] is a bitmask of x such that {a,b,x} is a bad triple
    # Indices 1..128, so we use 128 bits.
    pair_conflicts = [[0] * (N + 1) for _ in range(N + 1)]
    
    # To speed up, we can iterate over triples directly
    # But N=128, triples ~ 350k.
    # For each triple (a,b,c), check if gcd(a,b)==gcd(b,c)==gcd(c,a)
    # If so, mark c in pair_conflicts[a][b], a in pair_conflicts[b][c], etc.
    
    # Optimization: Group by gcd value?
    # Or just brute force precomputation.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                g_bc = gcd(b, c)
                if g_ab != g_bc:
                    continue
                g_ca = gcd(c, a)
                if g_ab == g_ca:
                    # Conflict found
                    pair_conflicts[a][b] |= (1 << c)
                    pair_conflicts[b][a] |= (1 << c)
                    
                    pair_conflicts[b][c] |= (1 << a)
                    pair_conflicts[c][b] |= (1 << a)
                    
                    pair_conflicts[c][a] |= (1 << b)
                    pair_conflicts[a][c] |= (1 << b)

    best_set = set()
    best_size = 0
    
    def can_add(x, S):
        # Check if x can be added to S
        # Iterate over pairs in S
        s_list = list(S)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                if pair_conflicts[a][b] & (1 << x):
                    return False
        return True

    # Randomized Greedy with Restarts
    for _ in range(5000):
        if time.time() - start_time > timeout:
            break
            
        current_set = set()
        candidates = numbers[:]
        random.shuffle(candidates)
        
        # Heuristic: Sort candidates by "conflict potential"
        # Potential = number of conflicts with current set?
        # Since set is empty initially, we can just pick randomly.
        # But as set grows, we should prefer numbers that don't conflict with many existing pairs.
        
        for x in candidates:
            if time.time() - start_time > timeout:
                break
            if can_add(x, current_set):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
            # Try local search (optional, might be slow)
            # Simple local search: try to swap elements
            # Given time constraints, simple greedy restarts are robust.
            
    return list(best_set)
# EVOLVE-BLOCK-END
