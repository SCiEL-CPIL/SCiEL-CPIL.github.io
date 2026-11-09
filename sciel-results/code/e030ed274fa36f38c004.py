import sys
import time
import math
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin before 90s kill
    
    N = 255
    target = 255
    
    # Filter candidates: must share a factor with 255
    candidates = [i for i in range(1, N + 1) if i != target and math.gcd(i, target) > 1]
    
    n = len(candidates)
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u in range(n):
        cu = candidates[u]
        for v in range(u + 1, n):
            cv = candidates[v]
            if math.gcd(cu, cv) > 1:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    best_clique_size = 0
    best_clique_bits = 0
    
    def bk(R_bits, P_bits, X_bits):
        nonlocal best_clique_size, best_clique_bits, deadline
        
        if time.time() > deadline:
            return
            
        # Count bits
        def popcount(x):
            c = 0
            while x:
                x &= x - 1
                c += 1
            return c
        
        r_size = popcount(R_bits)
        p_size = popcount(P_bits)
        x_size = popcount(X_bits)
        
        if p_size == 0 and x_size == 0:
            if r_size > best_clique_size:
                best_clique_size = r_size
                best_clique_bits = R_bits
            return
            
        if p_size == 0:
            if r_size > best_clique_size:
                best_clique_size = r_size
                best_clique_bits = R_bits
            return

        # Select pivot: node in P∪X with max neighbors in P
        PX = P_bits | X_bits
        best_pivot = -1
        max_deg = -1
        
        # Iterate through bits of PX
        temp = PX
        while temp:
            lb = temp & (-temp)
            u = lb.bit_length() - 1
            temp ^= lb
            
            # Count neighbors of u in P
            deg = popcount(adj[u] & P_bits)
            if deg > max_deg:
                max_deg = deg
                best_pivot = u
                
        if best_pivot == -1:
            return
            
        # Branch on P \ N(pivot)
        to_branch = P_bits & ~adj[best_pivot]
        
        temp = to_branch
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            temp ^= lb
            
            new_R = R_bits | lb
            new_P = P_bits & adj[v] & ~lb
            new_X = X_bits & adj[v]
            
            bk(new_R, new_P, new_X)
            
            P_bits &= ~lb
            X_bits |= lb
            
            if time.time() > deadline:
                return

    # Initial call
    initial_P = (1 << n) - 1
    bk(0, initial_P, 0)
    
    # Build result
    if best_clique_bits:
        result_values = []
        for i in range(n):
            if (best_clique_bits >> i) & 1:
                result_values.append(candidates[i])
        result_values.append(target)
        return result_values
    else:
        # Fallback: all multiples of 3
        return list(range(3, N + 1, 3))
# EVOLVE-BLOCK-END
