"""Program for R(K2,2,2) maximization using bitmask constraint satisfaction."""
import numpy as np

def construct():
    # We want to maximize n such that there exists a 2-coloring of K_n
    # with no monochromatic K_{2,2,2}.
    
    # Efficient check using bitmasks
    def has_k222(colors, n, color):
        """Check if color class contains K_{2,2,2} using bitmasks."""
        # Build adjacency bitmasks for the given color
        # adj[i] is a bitmask of neighbors of i in this color
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == color:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        
        # Check for K_{2,2,2}
        # K_{2,2,2} exists iff there is an edge (u,v) in color such that
        # N(u) & N(v) contains a K_{2,2}.
        # A set S contains K_{2,2} iff there exist a,b in S and c,d in S\{a,b}
        # such that a,b are connected to c,d.
        # This is equivalent to: there exist a,b in S such that
        # (N(a) & N(b) & S) has size >= 2 AND the vertices in that intersection
        # are NOT a and b (which is true by definition of common neighbors)
        # Wait, if a,b in S, and c,d in S, and a~c, a~d, b~c, b~d.
        # Then c,d are in N(a) & N(b). Also c,d must be in S.
        # So we need to check if for some a,b in S, |N(a) & N(b) & S| >= 2.
        # Note: N(a) & N(b) does not contain a or b (assuming no self-loops).
        
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    # u and v are connected in this color
                    common = adj[u] & adj[v]
                    # common contains vertices w such that u~w and v~w
                    # We need to check if 'common' (as a set of vertices) contains a K_{2,2}.
                    # Let S = common.
                    # We need to check if there exist a,b in S such that |N(a) & N(b) & S| >= 2.
                    # Iterate over pairs in S?
                    # To be faster:
                    # If S has < 4 vertices, it cannot contain K_{2,2,2} as the "other" part?
                    # No, K_{2,2,2} requires 6 vertices total. u,v are 2. S must have at least 4.
                    if common.bit_count() < 4:
                        continue
                    
                    # Check if S contains K_{2,2}
                    # We can iterate over pairs (a,b) in S.
                    # There are at most C(27, 2) ~ 351 pairs.
                    # For each pair, check intersection size.
                    s_list = []
                    mask = common
                    while mask:
                        lsb = mask & -mask
                        idx_w = lsb.bit_length() - 1
                        s_list.append(idx_w)
                        mask ^= lsb
                    
                    m = len(s_list)
                    # Optimization: if m is small, just do it.
                    # For each pair a, b in s_list:
                    found = False
                    for i in range(m):
                        a = s_list[i]
                        for j in range(i+1, m):
                            b = s_list[j]
                            # Check if a and b are connected in this color?
                            # Not strictly required for K_{2,2} inside S?
                            # K_{2,2} in S means parts {a,b} and {c,d} are fully connected.
                            # It does NOT require a~b.
                            # So we just need |N(a) & N(b) & S| >= 2.
                            # But wait, c and d must be distinct from a and b.
                            # N(a) & N(b) does not contain a or b.
                            # So we just need the intersection with S to have size >= 2.
                            inter = adj[a] & adj[b] & common
                            if inter.bit_count() >= 2:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return True
        return False

    def check_valid(colors, n):
        """Check if both color classes are K_{2,2,2}-free."""
        return not has_k222(colors, n, 0) and not has_k222(colors, n, 1)

    # Try Paley graphs for n=29, 31
    # Paley graph on q vertices (q prime power, q = 1 mod 4)
    # Edge (i,j) if (j-i) is a quadratic residue mod q.
    
    def get_paley_colors(q):
        squares = set()
        for x in range(1, q):
            squares.add((x*x) % q)
        # Note: 0 is not a square in the multiplicative group sense for this construction usually, 
        # but (j-i) can be 0? No i!=j.
        # The set of non-zero squares.
        colors = []
        for i in range(q):
            for j in range(i+1, q):
                diff = (j - i) % q
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Test n=29
    n = 29
    colors = get_paley_colors(n)
    if check_valid(colors, n):
        return {'n': n, 'colors': colors}
        
    # Test n=31 (just in case, though 31 is 3 mod 4, so Paley graph is not defined in the standard symmetric way? 
    # Actually Paley graphs exist for q=1 mod 4. For q=3 mod 4, the graph is not symmetric? 
    # (j-i) is square iff (i-j) is square? 
    # If q=3 mod 4, -1 is non-square. So (i-j) = -(j-i). 
    # If j-i is square, i-j is non-square. So the graph is not undirected.
    # So we cannot use standard Paley for 31.
    # We can use a random search or a different construction for 30/31 if 29 fails, 
    # but 29 is the known record.
    
    # If 29 fails (which it shouldn't), try random search for 29
    import random
    random.seed(42)
    for attempt in range(100):
        colors = [random.choice([0, 1]) for _ in range(n*(n-1)//2)]
        if check_valid(colors, n):
            return {'n': n, 'colors': colors}
            
    # Fallback to smaller n if 29 fails
    for n in [27, 25, 23, 17]:
        if n % 4 == 1:
            colors = get_paley_colors(n)
            if check_valid(colors, n):
                return {'n': n, 'colors': colors}
                
    # Absolute fallback
    return {'n': 1, 'colors': []}
