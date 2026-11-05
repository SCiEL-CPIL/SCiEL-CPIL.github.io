import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85
    n = 128
    nums = list(range(1, n + 1))
    
    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    def is_bad_triple(a, b, c):
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]

    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        s_list = list(S)
        m = len(s_list)
        for i in range(m):
            a = s_list[i]
            g_ax = gcd_table[a][x]
            for j in range(i + 1, m):
                b = s_list[j]
                # If gcd(a,b) == gcd(a,x) == gcd(b,x), it's a bad triple
                if gcd_table[a][b] == g_ax and gcd_table[b][x] == g_ax:
                    return False
        return True

    def greedy_build(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
        return S

    best_size = 0
    best_set = set()
    
    # Run multiple randomized greedy passes
    # Each pass is fast enough to run many times
    num_restarts = 5000
    
    for i in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        random.shuffle(nums)
        # We can also try sorting by some heuristic, but random shuffle is a good baseline
        # Maybe mix in some heuristic orders too?
        # For now, just random shuffle
        
        S = greedy_build(nums)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            # If we found a very large set, we can reduce remaining restarts or just keep going
            # Given the time budget, we can just keep going until deadline.
            
    # Final local search on the best set found
    # Try to add any remaining numbers
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for x in range(1, n + 1):
            if x not in best_set:
                if can_add(x, best_set):
                    best_set.add(x)
                    improved = True
                    break
        
        # If no single addition works, try a simple swap to see if it unlocks something?
        # Swaps keep size same, but might change structure.
        # Given the small N, we can try all swaps if time permits.
        if not improved:
             # Try swaps
             # This is O(N^2) checks, each O(|S|^2). 128^2 * 10000 = 1.6e9 ops. Too slow for Python in 1s.
             # Skip swaps for now, or do limited swaps.
             pass

    return list(best_set)
