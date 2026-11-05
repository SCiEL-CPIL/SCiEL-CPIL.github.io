import random
import time

def construct():
    """
    Attempts to find a valid 2-coloring of K_n with no monochromatic K_{2,2,2}.
    Uses Random Local Search with a fast bitset-based violation checker.
    Targets n=29, the known lower bound for R(K_{2,2,2}).
    """
    start_time = time.time()
    time_limit = 110  # Leave buffer for overhead
    
    def is_valid(n, colors):
        """
        Checks if the coloring of K_n (colors list of length n*(n-1)/2) 
        contains a monochromatic K_{2,2,2}.
        Returns True if valid (no mono K_{2,2,2}), False otherwise.
        """
        # Build adjacency bitsets for color 0 and 1
        # adj_c[i] is a bitmask of vertices j such that edge (i,j) has color c
        adj0 = [0] * n
        adj1 = [0] * n
        
        # Map edge index to (i, j)
        # We need to populate adj0/adj1 from the flat colors list
        # colors[k] corresponds to pair (i, j) in lexicographic order
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                k += 1
        
        def has_mono_k222(adj):
            """
            Checks if graph defined by adj bitsets contains K_{2,2,2}.
            K_{2,2,2} vertices: a1, a2 (A), b1, b2 (B), c1, c2 (C).
            Edges: All between A-B, A-C, B-C.
            Condition:
            Exists u, v (edge in G) such that:
            Common neighbors x, y exist in G (connected to both u and v).
            Exists a' connected to v, x, y.
            Exists b' connected to u, x, y.
            """
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        continue
                    
                    # Common neighbors of u and v in G
                    common = adj[u] & adj[v]
                    # Remove u and v themselves from consideration if they were in common (they are not, as no self loops)
                    # But we need distinct x, y
                    # Iterate over pairs x, y in common
                    # To be efficient, just check if popcount(common) >= 2
                    if common & (common - 1) == 0: # 0 or 1 bits set
                        continue
                    
                    # We need to check if there exist x, y in common 
                    # and a' in N(v) \cap N(x) \cap N(y)
                    # and b' in N(u) \cap N(x) \cap N(y)
                    # Note: a' and b' must be distinct from u, v, x, y
                    
                    # Iterate over all pairs x, y in common?
                    # This might be slow if common is large.
                    # However, in a random graph, common size is small.
                    # In a valid coloring, violations are rare.
                    
                    # Optimization: 
                    # For each x in common:
                    #   For each y in common where y > x:
                    #      candidates_a = adj[v] & adj[x] & adj[y]
                    #      candidates_b = adj[u] & adj[x] & adj[y]
                    #      Remove u, v, x, y from candidates
                    #      If candidates_a != 0 and candidates_b != 0:
                    #          return True
                    
                    # To avoid O(M^2) where M is popcount(common), we can be smarter?
                    # For n=29, max popcount is 27. 27*27/2 ~ 350 checks.
                    # Each check is 3 bitwise ANDs. Very fast.
                    
                    # Get list of vertices in common
                    # Using bit manipulation to iterate set bits
                    common_bits = common
                    verts = []
                    while common_bits:
                        lsb = common_bits & (-common_bits)
                        idx = (lsb.bit_length() - 1)
                        verts.append(idx)
                        common_bits ^= lsb
                    
                    for i_x in range(len(verts)):
                        x = verts[i_x]
                        for i_y in range(i_x + 1, len(verts)):
                            y = verts[i_y]
                            
                            # Mask to exclude u, v, x, y
                            mask_excl = ~((1 << u) | (1 << v) | (1 << x) | (1 << y))
                            
                            # a' must be neighbor of v, x, y
                            cand_a = adj[v] & adj[x] & adj[y] & mask_excl
                            if cand_a == 0:
                                continue
                            
                            # b' must be neighbor of u, x, y
                            cand_b = adj[u] & adj[x] & adj[y] & mask_excl
                            if cand_b == 0:
                                continue
                            
                            # We found a K_{2,2,2}
                            return True
            return False

        if has_mono_k222(adj0):
            return False
        if has_mono_k222(adj1):
            return False
        return True

    def find_violation_edge(n, colors):
        """
        Finds one edge involved in a monochromatic K_{2,2,2}.
        Returns (u, v) or None if valid.
        """
        adj0 = [0] * n
        adj1 = [0] * n
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                k += 1
        
        def find_in_adj(adj):
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        continue
                    common = adj[u] & adj[v]
                    if common & (common - 1) == 0:
                        continue
                    common_bits = common
                    verts = []
                    while common_bits:
                        lsb = common_bits & (-common_bits)
                        idx = (lsb.bit_length() - 1)
                        verts.append(idx)
                        common_bits ^= lsb
                    for i_x in range(len(verts)):
                        x = verts[i_x]
                        for i_y in range(i_x + 1, len(verts)):
                            y = verts[i_y]
                            mask_excl = ~((1 << u) | (1 << v) | (1 << x) | (1 << y))
                            cand_a = adj[v] & adj[x] & adj[y] & mask_excl
                            if cand_a == 0: continue
                            cand_b = adj[u] & adj[x] & adj[y] & mask_excl
                            if cand_b == 0: continue
                            return (u, v)
            return None
        
        res0 = find_in_adj(adj0)
        if res0:
            return res0
        res1 = find_in_adj(adj1)
        if res1:
            return res1
        return None

    def edge_index(n, u, v):
        if u > v:
            u, v = v, u
        # Number of edges before row u
        # Row i has n-1-i edges
        # Sum_{j=0}^{u-1} (n-1-j)
        # = u*(n-1) - u*(u-1)/2
        # Plus offset in row u: v - u - 1
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    # Try n=29 first
    for n in range(29, 16, -1):
        # Try a few random restarts
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize random coloring
            colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            
            # Local Search
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                
                viol = find_violation_edge(n, colors)
                if viol is None:
                    return {'n': n, 'colors': colors}
                
                u, v = viol
                idx = edge_index(n, u, v)
                # Flip the color
                colors[idx] ^= 1
                
            # If we didn't find a valid one in this attempt, try again
            # Optionally, we could keep the best state, but random restart is simpler

    # Fallback: Return a known valid small n if all else fails (should not happen for n>17)
    # The seed provided n=17. We should at least match it.
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
