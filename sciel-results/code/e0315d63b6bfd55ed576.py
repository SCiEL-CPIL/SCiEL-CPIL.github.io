import time
import random

def construct():
    # We aim for n=17. If we find a valid coloring, we return it.
    # If not, we fall back to n=16, etc.
    # The seed program tries random colorings. We use a backtracking search.
    
    target_n = 17
    if not _solve(target_n):
        target_n = 16
        if not _solve(target_n):
            target_n = 15
            if not _solve(target_n):
                return {'n': 1, 'colors': []}
                
    return {'n': target_n, 'colors': _colors}

def _solve(n):
    global _colors
    _colors = [0] * (n*(n-1)//2)
    
    # Precompute edge indices
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
            
    # Bitsets for adjacency of each color
    # adj[c][u] is a bitmask of neighbors of u in color c
    adj = [[0] * n for _ in range(4)]
    
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i*n - (i+1)//2 + (j - i - 1) # Formula for lexicographic index? 
        # Let's just use a 2D array or map for simplicity, or calculate correctly.
        # Index for (i, j) with i < j:
        # Sum of (n-1) + (n-2) + ... + (n-i) + (j-i-1)
        # Actually, simpler:
        # idx = 0
        # for k in range(i): idx += n - 1 - k
        # idx += j - i - 1
        
    # Let's use a 2D list for colors to make it easy, then flatten at the end.
    color_mat = [[-1]*n for _ in range(n)]
    
    # Heuristic: Order edges to maximize constraints early? 
    # For now, standard order is fine with good pruning.
    
    # Fast C6 check for a specific color
    def has_c6_color(c):
        # Check if adj[c] contains a C6
        # For each u, for each v in N(u), for each w in N(v)\{u}, ...
        # This is O(N^5) or O(N^6) worst case.
        # Optimization: For each pair (u, v) that is an edge, check if there is a path of length 4 from v back to u not using u.
        # Actually, just check all 6-cycles.
        # Since n is small (<=17), we can use a simple loop.
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[c][u] >> v) & 1:
                    continue
                # We have edge u-v. We need a path v -> ... -> u of length 4.
                # Let's find all paths of length 4 from v to u.
                # v -> w1 -> w2 -> w3 -> u
                # w1 != u, w2 != u, w3 != u, all distinct.
                
                # Get neighbors of v
                neigh_v = [x for x in range(n) if (adj[c][v] >> x) & 1 and x != u]
                for w1 in neigh_v:
                    # Get neighbors of w1
                    neigh_w1 = [x for x in range(n) if (adj[c][w1] >> x) & 1 and x != v and x != u]
                    for w2 in neigh_w1:
                        # Get neighbors of w2
                        neigh_w2 = [x for x in range(n) if (adj[c][w2] >> x) & 1 and x != w1 and x != u]
                        for w3 in neigh_w2:
                            # Check if w3 is connected to u
                            if (adj[c][w3] >> u) & 1:
                                # Check distinctness: u, v, w1, w2, w3
                                # w1 != u, v (by construction)
                                # w2 != v, u, w1 (by construction)
                                # w3 != w1, u, w2 (by construction)
                                # w3 != v? 
                                # If w3 == v, then v-w1-w2-v is a triangle, not part of C6 with u-v?
                                # The cycle is u-v-w1-w2-w3-u.
                                # Vertices: u, v, w1, w2, w3.
                                # We need 6 distinct vertices.
                                # Here we have 5 vertices u, v, w1, w2, w3.
                                # Wait, a C6 has 6 vertices.
                                # My path v->w1->w2->w3->u has length 4 edges, so 5 vertices including v and u.
                                # Total vertices in cycle: u, v, w1, w2, w3. That's 5 vertices.
                                # I missed one vertex.
                                # Path of length 4 from v to u: v-w1-w2-w3-w4-u.
                                # Let's re-evaluate.
                                pass
        return False
