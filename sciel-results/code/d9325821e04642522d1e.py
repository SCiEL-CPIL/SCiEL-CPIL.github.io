import time
import random

def construct():
    q = 29
    squares = {x * x % q for x in range(1, q)}
    
    n_base = q
    colors_base = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % q
            if diff in squares:
                colors_base.append(0)
            else:
                colors_base.append(1)
                
    # Build adjacency bitmasks for base graph
    adj = [[0] * n_base for _ in range(2)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = colors_base[idx]
            idx += 1
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
    # Add vertex n_base
    n_new = n_base + 1
    
    # Precompute: for each pair (i,j) in base, and for each color, 
    # check if adding vertex n_base with certain colors creates K_{2,2,2}
    
    # A K_{2,2,2} containing the new vertex v has 5 other vertices from base
    # Partitioned into 3 pairs, with v connected to all 5 in some color pattern
    
    # Actually, K_{2,2,2} has 6 vertices. If one is the new vertex, the other 5 are from base.
    # The new vertex must be in one of the three parts.
    # Case 1: new vertex is in part 1 (size 2). Then it's paired with one base vertex a.
    #   The other two parts have 2 base vertices each: {b,c} and {d,e}
    #   Edges: new-a (color c1), new-b, new-c, new-d, new-e (all color c1)
    #   Plus: a-b, a-c, a-d, a-e (color c1), b-c, d-e (color c1), b-d, b-e, c-d, c-e (color c2 or c1)
    #   Wait, in K_{2,2,2}, ALL cross edges between different parts are present.
    #   So if new is in part 1 with a, then new connects to b,c,d,e in color c1,
    #   a connects to b,c,d,e in color c1, and b-c, d-e are within parts (not in K_{2,2,2}),
    #   but b-d, b-e, c-d, c-e are cross edges between parts 2 and 3, so color c1.
    
    # Let me define: new vertex v, color assignment to edges (v,i) for i in base.
    # For a K_{2,2,2} containing v:
    # v is in one of 3 parts. WLOG, part 1 = {v, a}, part 2 = {b, c}, part 3 = {d, e}
    # All edges between different parts must have the SAME color (the color of that K_{2,2,2})
    # Edges: v-a, v-b, v-c, v-d, v-e, a-b, a-c, a-d, a-e, b-d, b-e, c-d, c-e all same color
    # Note: b-c is within part 2, d-e is within part 3, these are NOT in K_{2,2,2}
    
    # So for a given color c, we need:
    # colors_new[v->a] = c, colors_new[v->b] = c, colors_new[v->c] = c, 
    # colors_new[v->d] = c, colors_new[v->e] = c
    # adj[c][a] has b,c,d,e
    # adj[c][b] has d,e
    # adj[c][c] has d,e
    # adj[c][d] has b,e
    # adj[c][e] has b,c
    
    # Precompute for each color c, and each pair of disjoint 2-sets {b,c}, {d,e} in base,
    # the set of vertices a such that adj[c][a] contains {b,c,d,e} and adj[c][b] contains {d,e}, etc.
    
    import itertools
    
    def precompute_k222_patterns(color):
        """For each color, find all potential K_{2,2,2} patterns in base that could be extended."""
        patterns = []
        # Iterate over all ways to choose 5 vertices from base
        # a, b, c, d, e where {b,c} and {d,e} are the other two parts
        for a in range(n_base):
            for b in range(n_base):
                if b == a: continue
                for c in range(n_base):
                    if c == a or c == b: continue
                    for d in range(n_base):
                        if d == a or d == b or d == c: continue
                        for e in range(n_base):
                            if e == a or e == b or e == c or e == d: continue
                            # Check if this forms a K_{2,2,2} pattern with new vertex
                            # v-a, v-b, v-c, v-d, v-e all color `color`
                            # a-b, a-c, a-d, a-e all color `color`
                            # b-d, b-e, c-d, c-e all color `color`
                            if not (adj[color][a] & (1 << b)): continue
                            if not (adj[color][a] & (1 << c)): continue
                            if not (adj[color][a] & (1 << d)): continue
                            if not (adj[color][a] & (1 << e)): continue
                            if not (adj[color][b] & (1 << d)): continue
                            if not (adj[color][b] & (1 << e)): continue
                            if not (adj[color][c] & (1 << d)): continue
                            if not (adj[color][c] & (1 << e)): continue
                            # Pattern: new vertex must connect to a,b,c,d,e in color `color`
                            patterns.append((a, b, c, d, e, color))
        return patterns
    
    patterns0 = precompute_k222_patterns(0)
    patterns1 = precompute_k222_patterns(1)
    
    # For a given color assignment to new vertex, check if any pattern is violated
    def check_patterns(colors_new):
        """Check if any K_{2,2,2} pattern is violated."""
        # Build mask of vertices connected to new vertex in each color
        mask0 = 0
        mask1 = 0
        for i in range(n_base):
            if colors_new[i] == 0:
                mask0 |= (1 << i)
            else:
                mask1 |= (1 << i)
        
        # Check color 0 patterns
        for (a, b, c, d, e, col) in patterns0:
            if col == 0:
                # New vertex must connect to a,b,c,d,e in color 0
                if not (mask0 & (1 << a)): continue
                if not (mask0 & (1 << b)): continue
                if not (mask0 & (1 << c)): continue
                if not (mask0 & (1 << d)): continue
                if not (mask0 & (1 << e)): continue
                return True  # Violation found
        
        # Check color 1 patterns
        for (a, b, c, d, e, col) in patterns1:
            if col == 1:
                if not (mask1 & (1 << a)): continue
                if not (mask1 & (1 << b)): continue
                if not (mask1 & (1 << c)): continue
                if not (mask1 & (1 << d)): continue
                if not (mask1 & (1 << e)): continue
                return True  # Violation found
        
        return False
    
    # Also need to check K_{2,2,2} NOT containing the new vertex
    # But the base graph is already valid, so any K_{2,2,2} in the full graph must use the new vertex
    
    # Greedy backtracking search
    start_time = time.time()
    deadline = 140.0
    
    # Symmetry breaking: fix first edge to color 0
    colors_new = [0] * n_base
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n_base:
            return not check_patterns(colors_new)
        
        for c in [0, 1]:
            colors_new[pos] = c
            if backtrack(pos + 1):
                return True
        
        return False
    
    colors_new[0] = 0  # Symmetry breaking
    found = backtrack(1)
    
    if found:
        final_colors = colors_base + colors_new
        return {'n': n_new, 'colors': final_colors}
    
    # If no solution found for n=30, return n=29
    return {'n': n_base, 'colors': colors_base}
