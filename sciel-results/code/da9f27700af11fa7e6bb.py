import sys
import random

def construct():
    """
    Constructs a 4-coloring of K_n without monochromatic triangles.
    Uses GF(49) for the base 49 vertices.
    Attempts to extend to n=50 by finding a valid assignment for the 50th vertex.
    """
    
    def gf49_mul(a, b):
        a0 = a % 7
        a1 = a // 7
        b0 = b % 7
        b1 = b // 7
        c0 = (a0 * b0 - 3 * a1 * b1) % 7
        c1 = (a0 * b1 + a1 * b0 - a1 * b1) % 7
        return c0 + 7 * c1

    def gf49_sub(a, b):
        a0 = a % 7
        a1 = a // 7
        b0 = b % 7
        b1 = b // 7
        c0 = (a0 - b0) % 7
        c1 = (a1 - b1) % 7
        return c0 + 7 * c1

    def pow_gf49(base, exp):
        res = 1
        b = base
        e = exp
        while e > 0:
            if e % 2 == 1:
                res = gf49_mul(res, b)
            b = gf49_mul(b, b)
            e //= 2
        return res

    # Find primitive element
    primitive = None
    for g in range(2, 49):
        if pow_gf49(g, 48) == 1:
            if pow_gf49(g, 24) != 1 and pow_gf49(g, 16) != 1 and pow_gf49(g, 12) != 1:
                primitive = g
                break
    
    if primitive is None:
        return _seed_construct()

    g = primitive
    powers = [1]
    for i in range(1, 48):
        powers.append(gf49_mul(powers[-1], g))
        
    log_map = {}
    for i, p in enumerate(powers):
        log_map[p] = i
        
    n_base = 49
    # Precompute base colors for 49 vertices
    diff_color = {}
    for d in range(1, 49):
        if d in log_map:
            diff_color[d] = log_map[d] % 4

    base_colors = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            d = gf49_sub(j, i)
            base_colors.append(diff_color[d])
            
    # Build adjacency lists for each color class in the base graph
    # G_c contains edges (u,v) if color(u,v) == c
    adj = {0: [[] for _ in range(n_base)], 1: [[] for _ in range(n_base)], 2: [[] for _ in range(n_base)], 3: [[] for _ in range(n_base)]}
    
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj[c][i].append(j)
            adj[c][j].append(i)
            idx += 1

    # Attempt to extend to n=50
    # We need to assign a color k_v to each vertex v in 0..48 for the edge (49, v)
    # such that for any u, v, if k_u == k_v == c, then color(u, v) != c.
    # This means the set of vertices assigned color c must be an independent set in G_c.
    
    # We can try to find such an assignment using a simple greedy/randomized approach.
    # Since the graph is highly symmetric, a random assignment might work if we check constraints.
    # Or we can use a backtracking search.
    
    # Given time constraints, a simple randomized greedy approach with restarts.
    max_restarts = 1000
    found_assignment = None
    
    for _ in range(max_restarts):
        # Randomly assign colors to 0..48
        k = [random.randint(0, 3) for _ in range(n_base)]
        
        # Check validity:
        # For each color c, check if any two vertices u, v with k[u]=c and k[v]=c have base_color(u,v) == c.
        valid = True
        for c in range(4):
            vertices_with_c = [v for v in range(n_base) if k[v] == c]
            # Check all pairs in vertices_with_c
            for i in range(len(vertices_with_c)):
                u = vertices_with_c[i]
                for j in range(i+1, len(vertices_with_c)):
                    v = vertices_with_c[j]
                    # Get base color of (u, v)
                    # We need to find the index in base_colors or recompute
                    # Recomputing is safer and fast enough for 49 vertices
                    d = gf49_sub(v, u)
                    bc = diff_color[d]
                    if bc == c:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            found_assignment = k
            break
            
    if found_assignment:
        # Construct the full color list for n=50
        colors = []
        # Edges involving vertex 49 (last vertex)
        # Lexicographic order: (0,1)..(0,48), (1,2)..(1,48), ..., (48,49)
        # Wait, the order is (i,j) for i<j.
        # Vertex 49 is the last one.
        # Pairs (i, 49) for i=0..48.
        # These are the last 49 entries in the list?
        # Total edges for n=50: 50*49/2 = 1225.
        # Total edges for n=49: 49*48/2 = 1176.
        # The first 1176 edges are (i,j) with i,j < 49.
        # The last 49 edges are (i, 49) for i=0..48.
        
        # So we append found_assignment to base_colors
        colors = base_colors + found_assignment
        
        return {'n': 50, 'colors': colors}
    else:
        return {'n': 49, 'colors': base_colors}

def _seed_construct():
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
