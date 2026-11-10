"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    n = 128
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Precompute lcm for all pairs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute conflict triples: for each pair (a,b), which x creates a bad triple
    # A triple (a,b,x) is bad if lcm(a,b) == lcm(a,x) == lcm(b,x)
    # For efficiency, precompute for each pair (a,b) the set of x that conflict
    # But that's O(n^3) = 2M entries, manageable
    
    # Better: for a given set S, to check if x can be added, check all pairs in S
    # Let's precompute: for each x, and each pair (a,b), whether (a,b,x) is a bad triple
    # conflict[a][b][x] = True if adding x to {a,b} creates a bad triple
    # That's 128^3 = 2M booleans, fine
    
    # Actually, let's think differently. For a set S, x can be added iff
    # for all pairs (a,b) in S: NOT (lcm(a,b) == lcm(a,x) == lcm(b,x))
    
    # Precompute bad_triple[a][b] = set of x such that (a,b,x) is bad
    # This is O(n^3) to build but O(1) per check during search
    bad_triple = {}
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            l_ab = lcm_table[a][b]
            bads = set()
            for x in range(1, n+1):
                if x == a or x == b:
                    continue
                if lcm_table[a][x] == l_ab and lcm_table[b][x] == l_ab:
                    bads.add(x)
            bad_triple[(a,b)] = bads
            bad_triple[(b,a)] = bads
    
    def can_add(S, x):
        """Check if x can be added to set S without creating a bad triple."""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                if x in bad_triple[(a,b)]:
                    return False
        return True
    
    def is_valid(S):
        """Check if set S has no bad triples."""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                for k in range(j+1, len(S_list)):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                        return False
        return True
    
    def greedy_solve(order):
        """Greedy: add elements in given order if they don't conflict."""
        S = set()
        for x in order:
            if x not in S:
                if can_add(S, x):
                    S.add(x)
        return S
    
    def local_search(S, max_iters=5000):
        """Try to improve S by adding, removing, or swapping elements."""
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iters += 1
            
            # Try adding each missing element
            for x in range(1, n+1):
                if x not in S:
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try removing one element and adding two
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline:
                    break
                a = S_list[i]
                S2 = S - {a}
                # Try to add two elements
                added = 0
                new_S = set(S2)
                for x in range(1, n+1):
                    if x not in new_S:
                        if can_add(new_S, x):
                            new_S.add(x)
                            added += 1
                            if added == 2:
                                break
                if len(new_S) > len(S):
                    S = new_S
                    improved = True
                    break
        
        return S
    
    # Try multiple strategies
    best_S = set()
    
    # Strategy 1: Ascending order
    S1 = greedy_solve(range(1, n+1))
    S1 = local_search(S1)
    if len(S1) > len(best_S):
        best_S = S1
    
    # Strategy 2: Descending order
    S2 = greedy_solve(range(n, 0, -1))
    S2 = local_search(S2)
    if len(S2) > len(best_S):
        best_S = S2
    
    # Strategy 3: Random restarts
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, n+1))
        random.shuffle(order)
        S = greedy_solve(order)
        S = local_search(S)
        if len(S) > len(best_S):
            best_S = S
    
    # Final verification
    if not is_valid(best_S):
        # Fallback to a known valid small set
        best_S = {1}
    
    if not best_S:
        best_S = {1}
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
