import sys
from itertools import combinations

# Increase recursion limit just in case
sys.setrecursionlimit(10000)

def construct():
    # P is global from the seed
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitsets
    # adj[i] is an integer where bit j is set if i is adjacent to j
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Bron-Kerbosch algorithm with pivoting using bitsets
    max_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique
        if not P and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        # Choose pivot u from P union X to minimize branching
        # We want u that maximizes |P & adj[u]| (to minimize |P \ N(u)|)
        # Or simply pick any u in P U X.
        # Standard optimization: pick u in P U X such that |P & N(u)| is max.
        # For simplicity and speed in Python, we can just pick the first one if sets are small,
        # but bitsets allow us to count bits quickly.
        
        U = P | X
        if U == 0:
            return
            
        # Find pivot u
        # We can iterate through bits in U
        # To optimize, we might just pick the first bit.
        # Let's pick the bit with the highest degree in P?
        # For N=24, just picking the first is fine.
        u_bit = U & -U  # Get lowest set bit
        u = u_bit.bit_length() - 1
        
        # Candidates: P & ~adj[u]
        # Note: adj[u] contains u? No, we didn't set self-loops.
        # But u is in P or X.
        # If u is in P, u is not in adj[u].
        # If u is in X, u is not in P.
        
        candidates = P & ~adj[u]
        
        while candidates:
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            
            bron_kerbosch(R | v_bit, P & adj[v], X & adj[v])
            
            P &= ~v_bit
            X |= v_bit
            candidates &= ~v_bit
            
    bron_kerbosch(0, (1 << n) - 1, 0)
    
    # Fallback if no clique found
    if not max_clique:
        max_clique = [0]
        
    # If max clique size is 1, find any edge
    if len(max_clique) < 2:
        for u, v in edges:
            max_clique = [u, v]
            break
            
    branches = list(max_clique)
    
    # Construct paths
    paths = []
    for u, v in combinations(branches, 2):
        paths.append([u, v])
        
    return {'branches': branches, 'paths': paths}

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}
