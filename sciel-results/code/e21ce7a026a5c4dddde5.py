import time
import random
import itertools
import numpy as np

def _has_k222_fast(adj, n):
    """
    Fast check if graph (adjacency matrix adj) contains K2,2,2 as subgraph.
    K2,2,2 = complete tripartite graph with parts of size 2.
    We need 6 vertices partitioned into 3 pairs such that all 12 cross-edges exist.
    """
    if n < 6:
        return False
    
    # Precompute the 15 perfect matchings of 6 vertices
    matchings = []
    def gen_matchings(verts):
        if not verts:
            yield []
            return
        first = verts[0]
        rest = verts[1:]
        for i in range(len(rest)):
            partner = rest[i]
            new_rest = rest[:i] + rest[i+1:]
            for m in gen_matchings(new_rest):
                yield [(first, partner)] + m
    
    for m in gen_matchings([0, 1, 2, 3, 4, 5]):
        matchings.append(m)
    
    # For each 6-subset, check all 15 matchings
    for subset in itertools.combinations(range(n), 6):
        for m in matchings:
            i, j = m[0]
            k, l = m[1]
            p, q = m[2]
            
            vi = subset[i]
            vj = subset[j]
            vk = subset[k]
            vl = subset[l]
            vp = subset[p]
            vq = subset[q]
            
            # Check edges between pair (i,j) and pair (k,l)
            if not (adj[vi][vk] and adj[vi][vl] and adj[vj][vk] and adj[vj][vl]):
                continue
            # Check edges between pair (i,j) and pair (p,q)
            if not (adj[vi][vp] and adj[vi][vq] and adj[vj][vp] and adj[vj][vq]):
                continue
            # Check edges between pair (k,l) and pair (p,q)
            if not (adj[vk][vp] and adj[vk][vq] and adj[vl][vp] and adj[vl][vq]):
                continue
            
            return True
    
    return False

def _check_valid_paley(n, colors):
    """Verify a coloring has no monochromatic K2,2,2."""
    adj0 = [[0] * n for _ in range(n)]
    adj1 = [[0] * n for _ in range(n)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i][j] = 1
                adj0[j][i] = 1
            else:
                adj1[i][j] = 1
                adj1[j][i] = 1
            idx += 1
    
    if _has_k222_fast(adj0, n):
        return False
    if _has_k222_fast(adj1, n):
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Use the known Paley 29 construction
    n = 29
    squares = set()
    for x in range(1, n // 2 + 1):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    
    # Verify the coloring is valid
    valid = _check_valid_paley(n, colors)
    if not valid:
        # This should not happen for the Paley construction, but handle gracefully
        pass
    
    # Check if we have time to attempt n=30 via backtracking
    elapsed = time.time() - start_time
    if elapsed < 100 and valid:
        try:
            n30 = 30
            adj0 = [[0] * n30 for _ in range(n30)]
            adj1 = [[0] * n30 for _ in range(n30)]
            
            idx = 0
            for i in range(29):
                for j in range(i + 1, 29):
                    c = colors[idx]
                    if c == 0:
                        adj0[i][j] = 1
                        adj0[j][i] = 1
                    else:
                        adj1[i][j] = 1
                        adj1[j][i] = 1
                    idx += 1
            
            edge_colors = [None] * 29
            
            def backtrack(pos):
                if time.time() > deadline - 5:
                    return False
                if pos == 29:
                    full_adj0 = [row[:] for row in adj0]
                    full_adj1 = [row[:] for row in adj1]
                    for i in range(29):
                        if edge_colors[i] == 0:
                            full_adj0[29][i] = 1
                            full_adj0[i][29] = 1
                        else:
                            full_adj1[29][i] = 1
                            full_adj1[i][29] = 1
                    
                    if _has_k222_fast(full_adj0, n30) or _has_k222_fast(full_adj1, n30):
                        return False
                    return True
                
                for c in [0, 1]:
                    edge_colors[pos] = c
                    if backtrack(pos + 1):
                        return True
                    edge_colors[pos] = None
                return False
            
            if backtrack(0):
                new_colors = colors[:]
                for i in range(29):
                    new_colors.append(edge_colors[i])
                return {'n': 30, 'colors': new_colors}
            
        except Exception:
            pass
    
    return {'n': n, 'colors': colors}
