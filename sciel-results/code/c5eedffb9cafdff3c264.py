"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _contains_k222_bitmask(neigh, n):
    """Check if graph with neighbor bitmasks contains K2,2,2 as a subgraph.
    K2,2,2 = octahedron = complete tripartite graph with 3 parts of size 2.
    Need disjoint a,b,c,d,e,f such that all edges between different parts exist.
    Equivalent: exist pair (a,b) with S=N(a)∩N(b), and pair (c,d) in S such that
    |N(c)∩N(d)∩S| >= 2 (giving e,f).
    """
    for a in range(n):
        for b in range(a + 1, n):
            S = neigh[a] & neigh[b]
            # S is disjoint from {a,b} since a not in N(a)
            if bin(S).count('1') < 4:
                continue
            # Get list of vertices in S
            c_list = [v for v in range(n) if S & (1 << v)]
            for ci in range(len(c_list)):
                c = c_list[ci]
                for di in range(ci + 1, len(c_list)):
                    d = c_list[di]
                    T = neigh[c] & neigh[d] & S
                    if bin(T).count('1') >= 2:
                        return True
    return False

def _check_valid(adj):
    """Check if neither color class contains K2,2,2."""
    n = adj.shape[0]
    
    # Build neighbor bitmasks for color 1 graph
    neigh1 = [0] * n
    for i in range(n):
        mask = 0
        row = adj[i]
        for j in range(n):
            if i != j and row[j] == 1:
                mask |= (1 << j)
        neigh1[i] = mask
    
    # Build neighbor bitmasks for color 0 graph (edge exists if adj[i][j]==0 for i!=j)
    neigh0 = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and adj[i][j] == 0:
                mask |= (1 << j)
        neigh0[i] = mask
    
    if _contains_k222_bitmask(neigh1, n):
        return False
    if _contains_k222_bitmask(neigh0, n):
        return False
    return True

def _paley_coloring(p):
    """Generate Paley coloring for prime p ≡ 1 (mod 4)."""
    squares = set(x * x % p for x in range(1, p))
    adj = np.zeros((p, p), dtype=np.int8)
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            c = 0 if (j - i) % p in squares else 1
            colors.append(c)
            adj[i][j] = c
            adj[j][i] = c
    return p, colors, adj

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 5 sec safety margin before 120s limit
    
    best_n = 1
    best_colors = [0]
    
    # Try Paley constructions for primes p ≡ 1 (mod 4) up to 31
    # These are: 5, 13, 17, 29
    paley_primes = [29, 17, 13, 5]
    
    for p in paley_primes:
        if time.time() > DEADLINE:
            break
        n, colors, adj = _paley_coloring(p)
        if _check_valid(adj):
            if n > best_n:
                best_n = n
                best_colors = colors
    
    # If no Paley worked, try a simple valid coloring
    if best_n <= 1:
        # n=4: K4 can be colored without K2,2,2 (K2,2,2 needs 6 vertices)
        # Any coloring of K4 is valid since K2,2,2 requires 6 vertices
        best_n = 4
        best_colors = [0] * 6  # all color 0
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
