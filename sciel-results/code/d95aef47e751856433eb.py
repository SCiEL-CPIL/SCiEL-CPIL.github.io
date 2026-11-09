import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 18
    # Use bitmasks for adjacency for speed
    adj_mask = [0] * n  # adj_mask[i] has bit j set if edge (i,j) exists
    
    def is_k4_free(u, v):
        # Check if adding edge (u,v) creates a K4
        # Need to check if any two common neighbors are connected
        common = adj_mask[u] & adj_mask[v] & ~((1 << u) | (1 << v))
        # Check if any pair in common is connected
        for x in range(n):
            if common & (1 << x):
                # Check if x is connected to any other common neighbor
                others = common & ~(1 << x)
                if adj_mask[x] & others:
                    return False
        return True

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    def largest_triangle_free():
        """Find the largest triangle-free induced subgraph size using branch and bound."""
        # Precompute: for each vertex, which other vertices form a triangle with it
        # Triangle (a,b,c) exists if adj[a][b] and adj[b][c] and adj[c][a]
        
        # For a set S to be triangle-free, no three vertices in S can form a triangle.
        
        # Strategy: try to find if there's a triangle-free set of size s, starting from high s
        # Use backtracking with pruning
        
        best = 0
        
        # Precompute triangle existence for triples
        # tri[a][b] = bitmask of c such that (a,b,c) is a triangle
        tri = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a+1, n):
                if adj_mask[a] & (1 << b):
                    # Find common neighbors
                    common = adj_mask[a] & adj_mask[b] & ~((1 << a) | (1 << b))
                    tri[a][b] = common
                    tri[b][a] = common
        
        def backtrack(available, current_size, current_set):
            """Try to extend current_set with vertices from available to maximize size while staying triangle-free."""
            nonlocal best
            if current_size + available.bit_count() <= best:
                return
            if current_size > best:
                best = current_size
            
            # Try adding vertices
            avail = available
            while avail:
                v = (avail & -avail).bit_length() - 1  # lowest set bit
                avail &= avail - 1  # clear lowest set bit
                
                # Check if v can be added: v must not form a triangle with any two in current_set
                # For each pair (a,b) in current_set, check if v is in tri[a][b]
                can_add = True
                cs = current_set
                while cs:
                    a = (cs & -cs).bit_length() - 1
                    cs &= cs - 1
                    cs2 = current_set & ~(1 << a)
                    while cs2:
                        b = (cs2 & -cs2).bit_length() - 1
                        cs2 &= cs2 - 1
                        if tri[a][b] & (1 << v):
                            can_add = False
                            break
                    if not can_add:
                        break
                
                if can_add:
                    new_set = current_set | (1 << v)
                    # New available: remove v and any vertex that would form triangle with v and some in new_set
                    new_avail = available & ~(1 << v)
                    # For each w in new_avail, check if w forms triangle with v and any in current_set
                    na = new_avail
                    while na:
                        w = (na & -na).bit_length() - 1
                        na &= na - 1
                        # Check if w forms triangle with v and any in current_set
                        cs3 = current_set
                        while cs3:
                            a = (cs3 & -cs3).bit_length() - 1
                            cs3 &= cs3 - 1
                            if tri[v][a] & (1 << w):
                                new_avail &= ~(1 << w)
                                break
                    backtrack(new_avail, current_size + 1, new_set)
        
        backtrack((1 << n) - 1, 0, 0)
        return best

    # Greedy construction with evaluation
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges)
    
    current_best_size = n  # Start with empty graph, max triangle-free is n
    
    for u, v in edges:
        if time.time() > deadline:
            break
        if not (adj_mask[u] & (1 << v)):
            if is_k4_free(u, v):
                add_edge(u, v)
                new_size = largest_triangle_free()
                if new_size < current_best_size:
                    current_best_size = new_size
                else:
                    remove_edge(u, v)
    
    # Local search: try to improve by adding edges that reduce max triangle-free size
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            if time.time() > deadline:
                break
            for v in range(u+1, n):
                if time.time() > deadline:
                    break
                if not (adj_mask[u] & (1 << v)):
                    if is_k4_free(u, v):
                        add_edge(u, v)
                        new_size = largest_triangle_free()
                        if new_size < current_best_size:
                            current_best_size = new_size
                            improved = True
                        else:
                            remove_edge(u, v)
    
    # Return edges
    res = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_mask[i] & (1 << j):
                res.append([i, j])
    return res
