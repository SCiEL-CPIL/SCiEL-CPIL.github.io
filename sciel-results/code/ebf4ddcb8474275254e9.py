# EVOLVE-BLOCK-START
import time
import numpy as np

def _contains_k222_bitmask(neigh):
    """Check if graph (given as list of neighbor bitmasks) contains K2,2,2."""
    n = len(neigh)
    for a in range(n):
        for b in range(a + 1, n):
            S = neigh[a] & neigh[b]
            if S.bit_count() < 4:
                continue
            verts = []
            mask = S
            while mask:
                v = (mask & (-mask)).bit_length() - 1
                verts.append(v)
                mask &= mask - 1
            L = len(verts)
            for i in range(L):
                c = verts[i]
                for j in range(i + 1, L):
                    d = verts[j]
                    T = neigh[c] & neigh[d] & S
                    if T.bit_count() >= 2:
                        return True
    return False

def _build_neigh(adj):
    n = adj.shape[0]
    neigh = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                neigh[i] |= (1 << j)
                neigh[j] |= (1 << i)
    return neigh

def _check_valid(adj):
    n = adj.shape[0]
    neigh1 = _build_neigh(adj)
    if _contains_k222_bitmask(neigh1):
        return False
    
    neigh0 = [0] * n
    for i in range(n):
        full_mask = (1 << n) - 1
        full_mask ^= (1 << i)
        neigh0[i] = full_mask ^ neigh1[i]
        
    if _contains_k222_bitmask(neigh0):
        return False
    return True

def _paley_coloring(p):
    squares = {x * x % p for x in range(1, p)}
    colors = []
    adj = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for j in range(i + 1, p):
            c = 1 if (j - i) % p in squares else 0
            colors.append(c)
            if c == 1:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj, colors

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    # 1. Paley 29
    adj29, colors29 = _paley_coloring(29)
    if _check_valid(adj29):
        best_result = {'n': 29, 'colors': colors29}
    else:
        best_result = {'n': 1, 'colors': [0]}
        
    # 2. Attempt to find n=30
    # We will try a few random extensions of Paley 29
    # and also some random 30-vertex graphs
    # Time budget: 10 seconds
    
    search_deadline = start_time + 10.0
    
    # Strategy: Start with Paley 29, add vertex 29, and try to satisfy constraints
    # We use a simple backtracking on the 29 edges to vertex 29
    # This is 2^29, so we need pruning.
    # Pruning: Check if the current partial assignment to vertex 29
    # creates a K2,2,2 in either color.
    
    # This is complex to implement efficiently in Python for 29 vars.
    # Instead, we try local search on a random 30-graph.
    
    # Given the low probability of random success, and the high cost of local search,
    # we will just return the best known (29) unless we can quickly verify a candidate.
    
    # Let's try one specific heuristic:
    # Take Paley 29, add vertex 29.
    # Color edges (29, i) based on some rule, e.g., all 0s, or alternating.
    # Then check validity.
    
    candidates = []
    
    # Candidate 1: All edges to 29 are 0
    adj30 = adj29.copy()
    # Resize adj29 to 30x30
    adj30 = np.zeros((30, 30), dtype=np.int8)
    adj30[:29, :29] = adj29
    for i in range(29):
        adj30[i, 29] = 0
        adj30[29, i] = 0
        
    if _check_valid(adj30):
        colors30 = []
        for i in range(30):
            for j in range(i + 1, 30):
                colors30.append(adj30[i, j])
        return {'n': 30, 'colors': colors30}
        
    # Candidate 2: All edges to 29 are 1
    for i in range(29):
        adj30[i, 29] = 1
        adj30[29, i] = 1
        
    if _check_valid(adj30):
        colors30 = []
        for i in range(30):
            for j in range(i + 1, 30):
                colors30.append(adj30[i, j])
        return {'n': 30, 'colors': colors30}
        
    # Candidate 3: Random
    np.random.seed(42)
    for _ in range(20):
        for i in range(29):
            adj30[i, 29] = np.random.randint(2)
            adj30[29, i] = adj30[i, 29]
            
        if _check_valid(adj30):
            colors30 = []
            for i in range(30):
                for j in range(i + 1, 30):
                    colors30.append(adj30[i, j])
            return {'n': 30, 'colors': colors30}
            
    return best_result

# EVOLVE-BLOCK-END
