import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def check_k222_free(n, colors):
        """
        Check if the graph defined by colors (1 for edge, 0 for non-edge) 
        contains K2,2,2 as a subgraph.
        Returns True if K2,2,2-free.
        """
        # Build adjacency list and bitsets for speed
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        
        # Precompute non-adjacency? 
        # We need to find 3 pairs (u,v), (x,y), (z,w) such that:
        # 1. u,v non-adjacent
        # 2. x,y non-adjacent
        # 3. z,w non-adjacent
        # 4. All cross edges exist.
        
        # Iterate over all non-edges (u,v) with u < v
        # For each, get common neighbors S = N(u) & N(v)
        # In S, we need to find two disjoint non-edges (x,y) and (z,w)
        # such that x,y,z,w form a K2,2 in the induced subgraph on S?
        # No, x,y must be non-adjacent in G.
        # z,w must be non-adjacent in G.
        # And x,y must be connected to z,w in G.
        
        # Let's iterate over pairs (x,y) in S that are non-adjacent.
        # For each such pair, let T = S \ {x,y} intersect N(x) & N(y)?
        # No, we just need to find (z,w) in S \ {x,y} such that:
        # - z,w non-adjacent
        # - z,w connected to x,y
        # - z,w connected to u,v (already in S)
        
        # So for fixed (u,v) non-edge:
        # S = N(u) & N(v)
        # Create list of vertices in S: s_list
        # For each pair (x,y) in s_list with x < y:
        #    if x,y are non-adjacent in G:
        #        # Check if there exists (z,w) in s_list \ {x,y}
        #        # such that z,w non-adjacent and z,w connected to x,y
        #        # This is equivalent to checking if the subgraph induced by 
        #        # (s_list \ {x,y}) contains a non-edge (z,w) where 
        #        # z,w are both in N(x) & N(y).
        #        # Let C = (N(x) & N(y)) & S  (excluding x,y is implicit if we check indices)
        #        # Actually, z,w must be in S.
        #        # So C = (N(x) & N(y)) & S
        #        # We need to find a non-edge in G[C].
        #        # If G[C] is not a clique, it has a non-edge.
        #        # So we just need to check if G[C] is a clique.
        #        # If it is NOT a clique, we found a K2,2,2.
        #        # Wait, we need z,w to be distinct and in C.
        #        # If |C| >= 2 and G[C] is not a clique, then there exist z,w in C 
        #        # that are non-adjacent.
        #        # Then {u,v}, {x,y}, {z,w} forms K2,2,2.
        
        for u in range(n):
            for v in range(u+1, n):
                # Check if u,v is a non-edge
                if (adj[u] >> v) & 1:
                    continue
                
                S = adj[u] & adj[v]
                # Get list of vertices in S
                s_list = [k for k in range(n) if (S >> k) & 1]
                
                # If |S| < 4, cannot form K2,2,2 with u,v as one part
                if len(s_list) < 4:
                    continue
                
                # Iterate over pairs (x,y) in s_list
                m = len(s_list)
                for i in range(m):
                    x = s_list[i]
                    for j in range(i+1, m):
                        y = s_list[j]
                        
                        # Check if x,y is a non-edge
                        if (adj[x] >> y) & 1:
                            continue
                        
                        # Find C = vertices in S \ {x,y} that are connected to both x and y
                        # C = S & N(x) & N(y)
                        # Since x,y in S, they are connected to u,v.
                        # We need z,w in C.
                        # Note: x is not in N(x), y is not in N(y).
                        # So C automatically excludes x and y?
                        # N(x) does not contain x. So x is not in N(x) & N(y).
                        # So C is disjoint from {x,y}.
                        
                        C = S & adj[x] & adj[y]
                        
                        # Check if there exist z,w in C such that z,w is a non-edge.
                        # This is true if G[C] is not a clique.
                        # Iterate over pairs in C
                        c_list = [k for k in range(n) if (C >> k) & 1]
                        
                        if len(c_list) < 2:
                            continue
                        
                        # Check if any pair in c_list is a non-edge
                        found_non_edge = False
                        for a in range(len(c_list)):
                            z = c_list[a]
                            for b in range(a+1, len(c_list)):
                                w = c_list[b]
                                if not ((adj[z] >> w) & 1):
                                    found_non_edge = True
                                    break
                            if found_non_edge:
                                break
                        
                        if found_non_edge:
                            return False # K2,2,2 found
        
        return True

    def is_valid(n, colors):
        """Check that both color classes are K2,2,2-free."""
        if not check_k222_free(n, colors):
            return False
        swapped = [1 - c for c in colors]
        if not check_k222_free(n, swapped):
            return False
        return True

    # Generate Paley(29)
    q = 29
    squares = set(x*x % q for x in range(1, q))
    colors_29 = []
    for i in range(q):
        for j in range(i+1, q):
            d = (j - i) % q
            if d in squares:
                colors_29.append(0)
            else:
                colors_29.append(1)

    # Verify quickly
    if is_valid(q, colors_29):
        return {'n': q, 'colors': colors_29}

    # Fallback: try Paley(17)
    q2 = 17
    squares2 = set(x*x % q2 for x in range(1, q2))
    colors_17 = []
    for i in range(q2):
        for j in range(i+1, q2):
            d = (j - i) % q2
            if d in squares2:
                colors_17.append(0)
            else:
                colors_17.append(1)

    if is_valid(q2, colors_17):
        return {'n': q2, 'colors': colors_17}

    # Fallback: try Paley(13)
    q3 = 13
    squares3 = set(x*x % q3 for x in range(1, q3))
    colors_13 = []
    for i in range(q3):
        for j in range(i+1, q3):
            d = (j - i) % q3
            if d in squares3:
                colors_13.append(0)
            else:
                colors_13.append(1)
                
    if is_valid(q3, colors_13):
        return {'n': q3, 'colors': colors_13}

    # Last resort: random small coloring
    for n_try in range(13, 5, -2):
        colors = [random.randint(0, 1) for _ in range(n_try*(n_try-1)//2)]
        if is_valid(n_try, colors):
            return {'n': n_try, 'colors': colors}

    return {'n': 5, 'colors': [0]*10}
