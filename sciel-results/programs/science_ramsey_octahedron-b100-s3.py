"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _check_valid_fast(adj):
    """Check if neither color class contains K2,2,2 as a subgraph.
    
    K2,2,2 is the octahedron graph: complete tripartite K_{2,2,2}.
    We check if any 6-vertex subset can be partitioned into 3 pairs such that
    all 12 cross-edges exist (i.e., the induced subgraph contains K2,2,2).
    
    Efficient approach: For each pair (a,b), let S = N(a)∩N(b).
    If |S| >= 4, check if there exist two disjoint pairs (c,d) and (e,f) in S
    such that all edges between {c,d} and {e,f} exist.
    This is equivalent to: does the subgraph induced by S contain K2,2?
    """
    n = adj.shape[0]
    
    # adj[i,j] = 1 means edge in color-1 graph
    # For color-0 graph, edge exists where adj[i,j] = 0 (and i != j)
    
    def _contains_k222(graph):
        # graph is n x n boolean/integer matrix, symmetric, no self-loops
        # Check if graph contains K2,2,2 as a subgraph
        # K2,2,2: exist 3 disjoint pairs {a,b},{c,d},{e,f} with all 12 cross-edges
        
        # Precompute neighbor bitmasks
        # n <= 31, so use Python int bitmasks
        neigh = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and graph[i][j]:
                    neigh[i] |= (1 << j)
        
        # For each pair (a,b), S = N(a) ∩ N(b)
        # We need to find two disjoint pairs (c,d) and (e,f) in S
        # such that c~e, c~f, d~e, d~f all hold.
        # Equivalently, the subgraph induced by S contains K2,2.
        
        for a in range(n):
            for b in range(a+1, n):
                S = neigh[a] & neigh[b]
                if bin(S).count('1') < 4:
                    continue
                
                # Get list of vertices in S
                verts = []
                mask = S
                while mask:
                    v = (mask & (-mask)).bit_length() - 1
                    verts.append(v)
                    mask &= mask - 1
                
                # Check if subgraph induced by verts contains K2,2
                # For each pair (c,d) in verts, T = N(c)∩N(d)∩S
                # If |T| >= 2 and T is disjoint from {c,d}, we have a K2,2
                # Actually c,d not in S automatically since c,d in S and S=N(a)∩N(b),
                # and a,b not in S. But c is in N(c)? No. So c not in N(c)∩N(d).
                # So T is automatically disjoint from {c,d}.
                
                L = len(verts)
                for i in range(L):
                    c = verts[i]
                    for j in range(i+1, L):
                        d = verts[j]
                        T = neigh[c] & neigh[d] & S
                        if bin(T).count('1') >= 2:
                            return True
        return False
    
    # Check color 1
    if _contains_k222(adj):
        return False
    
    # Check color 0: edge exists where adj=0 and i!=j
    # Build complement graph
    comp = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 0:
                comp[i][j] = 1
    
    if _contains_k222(comp):
        return False
    
    return True

def _paley_coloring(p):
    """Generate Paley coloring for prime p ≡ 1 (mod 4)."""
    squares = {x * x % p for x in range(1, p)}
    adj = np.zeros((p, p), dtype=np.int8)
    colors = []
    for i in range(p):
        for j in range(i+1, p):
            c = 1 if (j - i) % p in squares else 0
            colors.append(c)
            if c == 1:
                adj[i][j] = 1
                adj[j][i] = 1
    return adj, colors

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0
    
    best_result = {'n': 1, 'colors': [0]}
    
    # Try Paley 29 (largest prime ≡ 1 mod 4 that is ≤ 31)
    if time.time() < DEADLINE:
        adj29, colors29 = _paley_coloring(29)
        if _check_valid_fast(adj29):
            best_result = {'n': 29, 'colors': colors29}
            return best_result
    
    # Try Paley 17
    if time.time() < DEADLINE:
        adj17, colors17 = _paley_coloring(17)
        if _check_valid_fast(adj17):
            best_result = {'n': 17, 'colors': colors17}
            return best_result
    
    # Try Paley 13
    if time.time() < DEADLINE:
        adj13, colors13 = _paley_coloring(13)
        if _check_valid_fast(adj13):
            best_result = {'n': 13, 'colors': colors13}
            return best_result
    
    # Try Paley 5
    if time.time() < DEADLINE:
        adj5, colors5 = _paley_coloring(5)
        if _check_valid_fast(adj5):
            best_result = {'n': 5, 'colors': colors5}
            return best_result
    
    return best_result

# EVOLVE-BLOCK-END
