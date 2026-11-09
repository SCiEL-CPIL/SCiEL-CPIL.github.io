import time
import random
import math
import numpy as np

def _has_c6_exact(adj_masks, n):
    """
    Check if any color class contains a C6.
    adj_masks: list of 4 lists, each of length n, containing bitmasks of neighbors.
    adj_masks[c][u] is an integer where bit v is 1 if edge (u,v) is color c.
    """
    for c in range(4):
        masks = adj_masks[c]
        # Precompute N(u) as sets or lists for iteration?
        # Bitmask iteration:
        for u in range(n):
            mask_u = masks[u]
            # Iterate over neighbors v of u
            v = 0
            while mask_u:
                # Get lowest set bit
                if mask_u & 1:
                    v = 0
                    # Actually, we need to iterate over set bits efficiently
                    pass
                mask_u >>= 1
                v += 1
            
            # Better iteration:
            m = masks[u]
            while m:
                # lsb = m & -m
                # v = lsb.bit_length() - 1
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                
                if u == v: continue
                # Check if there is a path of length 4 from u to v in G_c \ {(u,v)}
                # Path: u-a-b-c-d-v
                # a in N(u) \ {v}
                # d in N(v) \ {u}
                # b in N(a)
                # c in N(b) \cap N(d)
                # And all vertices distinct: u,a,b,c,d,v distinct.
                
                # Get neighbors of u excluding v
                neighbors_u = masks[u] & ~(1 << v)
                
                # Get neighbors of v excluding u
                neighbors_v = masks[v] & ~(1 << u)
                
                # For each a in neighbors_u:
                ma = neighbors_u
                while ma:
                    lsb_a = ma & -ma
                    a = lsb_a.bit_length() - 1
                    ma ^= lsb_a
                    
                    # a must be distinct from u, v (guaranteed by mask)
                    # b in N(a)
                    # b must be distinct from u, v, a
                    # c in N(b) \cap N(d)
                    # d in neighbors_v
                    # c must be distinct from u, v, a, b, d
                    
                    # This is getting complex. Let's use a simpler check for small N.
                    # Since N <= 20, we can just do a BFS/DFS of depth 4?
                    # Or simply check all paths u-a-b-c-d-v.
                    
                    # Let's stick to a slightly slower but simpler exact check using sets if bitmasks are too complex to manage distinctness.
                    # For N=17, sets are fast enough if called infrequently.
                    pass
                
    return False

def _check_validity(n, colors):
    """
    Exact check if the coloring is valid (no C6 in any color).
    """
    # Build adjacency lists for each color
    adj = [set() for _ in range(4)]
    for i in range(n):
        adj[i] = [set() for _ in range(4)]
        
    for i in range(n):
        for j in range(i+1, n):
            idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
            # Correct index calculation:
            # Number of edges before row i: i*(n-1) - i*(i-1)/2 ?
            # Row 0: n-1 edges. Index 0..n-2.
            # Row 1: n-2 edges. Index n-1 .. 2n-3.
            # Formula: i*(n-1) - i*(i-1)//2 + (j-i-1)
            # Let's verify for i=0, j=1: 0 + 0 = 0. Correct.
            # i=0, j=n-1: n-2. Correct.
            # i=1, j=2: 1*(n-1) - 0 + 0 = n-1. Correct.
            
            c = colors[idx]
            adj[i][c].add(j)
            adj[j][c].add(i)
            
    # Check for C6 in each color
    for c in range(4):
        for u in range(n):
            for v in adj[u][c]:
                if u >= v: continue
                # Check if path of length 4 exists from u to v in G_c \ {(u,v)}
                # Path u-a-b-c-d-v
                # a in adj[u][c] \ {v}
                # d in adj[v][c] \ {u}
                # b in adj[a][c]
                # c_node in adj[b][c] \cap adj[d][c]
                # Distinctness: u,a,b,c_node,d,v all distinct.
                
                found = False
                for a in adj[u][c]:
                    if a == v: continue
                    for d in adj[v][c]:
                        if d == u: continue
                        if d == a: continue # a and d must be distinct
                        # Now find b, c_node
                        # b in adj[a][c] \ {u, v, a, d}
                        # c_node in adj[b][c] \ {u, v, a, b, d}
                        # and c_node in adj[d][c]
                        for b in adj[a][c]:
                            if b in (u, v, a, d): continue
                            for c_node in adj[b][c]:
                                if c_node in (u, v, a, b, d): continue
                                if c_node in adj[d][c]:
                                    found = True
                                    break
                            if found: break
                        if found: break
                    if found: break
                if found:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try n=17 first, then 16, 15...
    for n in [17, 16, 15, 14]:
        if time.time() > deadline - 5:
            break
            
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Helper to get edge index
        def edge_idx(i, j):
            if i > j: i, j = j, i
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
            
        # Build initial adjacency matrices for NumPy
        # adj_matrices[c] is n x n numpy array
        adj_matrices = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_idx(i, j)]
                adj_matrices[c][i][j] = 1
                adj_matrices[c][j][i] = 1
        
        def get_surrogate_cost():
            total = 0
            for c in range(4):
                # Trace of A^6
                # For small n, matrix_power is fine
                try:
                    A6 = np.linalg.matrix_power(adj_matrices[c], 6)
                    total += np.trace(A6)
                except:
                    pass
            return total
        
        initial_cost = get_surrogate_cost()
        best_cost = initial_cost
        best_colors = colors.copy()
        
        T = 1.0
        cooling = 0.999
        steps = 0
        max_steps = 500000
        check_interval = 500
        
        while time.time() < deadline - 5 and steps < max_steps:
            steps += 1
            
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            idx = edge_idx(i, j)
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            if old_c == new_c:
                continue
            
            # Update matrices
            adj_matrices[old_c][i][j] = 0
            adj_matrices[old_c][j][i] = 0
            adj_matrices[new_c][i][j] = 1
            adj_matrices[new_c][j][i] = 1
            
            new_cost = get_surrogate_cost()
            delta = new_cost - best_cost
            
            # Accept if better or with prob exp(-delta/T)
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[idx] = new_c
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors.copy()
                    
                # Periodic exact check
                if steps % check_interval == 0 and best_cost < 1000:
                    if _check_validity(n, best_colors):
                        return {'n': n, 'colors': best_colors}
            else:
                # Revert
                adj_matrices[old_c][i][j] = 1
                adj_matrices[old_c][j][i] = 1
                adj_matrices[new_c][i][j] = 0
                adj_matrices[new_c][j][i] = 0
                
            T *= cooling
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
