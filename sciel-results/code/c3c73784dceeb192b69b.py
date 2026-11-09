import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Fallback: Greenwood-Gleason K32
    def fallback_construct():
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

    try:
        p = 7
        # GF(49) elements as (a, b) representing a + b*x
        elements = [(a, b) for b in range(p) for a in range(p)]
        
        def add(e1, e2):
            return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
        
        def sub(e1, e2):
            return ((e1[0] - e2[0]) % p, (e1[1] - e2[1]) % p)
        
        def mul(e1, e2):
            a1, b1 = e1
            a2, b2 = e2
            # x^2 = 6x + 4
            a_res = (a1 * a2 + 4 * b1 * b2) % p
            b_res = (a1 * b2 + a2 * b1 + 6 * b1 * b2) % p
            return (a_res, b_res)
            
        def pow_el(e, k):
            res = (1, 0)
            base = e
            while k > 0:
                if k & 1:
                    res = mul(res, base)
                base = mul(base, base)
                k >>= 1
            return res
            
        def is_primitive(e):
            if e == (0, 0): return False
            # Order 48. Factors 2, 3.
            if pow_el(e, 24) == (1, 0): return False
            if pow_el(e, 16) == (1, 0): return False
            return True
            
        primitive = None
        for e in elements:
            if e != (0, 0) and is_primitive(e):
                primitive = e
                break
                
        if primitive is None:
            return fallback_construct()
            
        # Generate multiplicative group
        group = []
        cur = (1, 0)
        for _ in range(48):
            group.append(cur)
            cur = mul(cur, primitive)
            
        # Check if subgroup of index 4 is sum-free
        # H = { g^(4k) }
        H = [group[i] for i in range(0, 48, 4)]
        
        # Check for zero-sum triple in H
        sum_free = True
        h_set = set(H)
        for a in H:
            for b in H:
                c = sub((0,0), add(a, b)) # c = -(a+b)
                if c in h_set:
                    sum_free = False
                    break
            if not sum_free:
                break
                
        if not sum_free:
            # Try other partitions? Or fallback
            return fallback_construct()
            
        # Partition group into 4 cosets
        # C_i = { g^(4k+i) }
        cosets = [[], [], [], []]
        for i in range(48):
            cosets[i % 4].append(group[i])
            
        # Map element to color
        color_map = {}
        for c_idx in range(4):
            for el in cosets[c_idx]:
                color_map[el] = c_idx
                
        # Build coloring for K49
        n = 49
        colors = []
        edge_color = [[-1]*n for _ in range(n)]
        
        for i in range(n):
            vi = elements[i]
            for j in range(i+1, n):
                vj = elements[j]
                diff = sub(vi, vj)
                if diff == (0, 0):
                    c = 0
                else:
                    c = color_map[diff]
                edge_color[i][j] = c
                edge_color[j][i] = c
                colors.append(c)
                
        # Verify K49
        valid = True
        for c in range(4):
            for i in range(n):
                for j in range(i+1, n):
                    if edge_color[i][j] != c: continue
                    for k in range(j+1, n):
                        if edge_color[i][k] == c and edge_color[j][k] == c:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid: break
            
        if not valid:
            return fallback_construct()
            
        # Try to extend to n=50
        # Add vertex 50 (index 49)
        # We need to assign colors to edges (49, j) for j=0..48
        # such that no monochromatic triangle is formed.
        # Triangle (49, j, k) is monochromatic color c if:
        # edge(49, j)=c, edge(49, k)=c, edge(j, k)=c.
        
        # This is equivalent to: For each color c, the set of vertices {j | edge(49, j)=c}
        # must be an independent set in the graph G_c (edges of color c).
        
        # We can try to construct this set.
        # Let S_c be the set of vertices assigned color c to the new vertex.
        # S_0, S_1, S_2, S_3 partition {0..48}.
        # Constraint: For each c, no edge of color c exists within S_c.
        
        # Greedy approach:
        # Initialize S_c as empty.
        # Assign each vertex j to a color c such that j can be added to S_c.
        # j can be added to S_c if no neighbor k of j in G_c is already in S_c.
        
        # Let's try a simple backtracking or greedy with randomization.
        # Given time limits, a few random restarts might work.
        
        best_n = 49
        best_colors = colors[:]
        
        for attempt in range(50):
            if time.time() > deadline - 5: break
            
            # Randomly shuffle vertex order
            vertices = list(range(49))
            import random
            random.shuffle(vertices)
            
            S = [[], [], [], []]
            assigned = [False] * 49
            possible = True
            
            for v in vertices:
                # Find available colors for v
                avail_colors = []
                for c in range(4):
                    # Check if v can be added to S_c
                    # v can be added if no neighbor u of v in G_c is in S_c
                    ok = True
                    for u in S[c]:
                        if edge_color[v][u] == c:
                            ok = False
                            break
                    if ok:
                        avail_colors.append(c)
                            
                if not avail_colors:
                    possible = False
                    break
                    
                # Pick a color, maybe randomly from available
                c = random.choice(avail_colors)
                S[c].append(v)
                assigned[v] = True
                
            if possible:
                # Construct new colors list
                new_colors = best_colors[:]
                # Append colors for edges (49, j)
                # Order must be lexicographic: (0,49), (1,49), ..., (48,49)
                # Wait, the list is for pairs (i,j) with i<j.
                # The new vertex is 49.
                # Pairs are (i, 49) for i=0..48.
                # These come after all pairs (i,j) with j<49.
                # So we just append.
                for i in range(49):
                    # Find which color i was assigned
                    for c in range(4):
                        if i in S[c]:
                            new_colors.append(c)
                            break
                            
                # Verify n=50
                n_new = 50
                valid_new = True
                # Only need to check triangles involving vertex 49
                # Triangles (49, i, j)
                for i in range(49):
                    for j in range(i+1, 49):
                        c_ij = edge_color[i][j]
                        # c_49_i is the color assigned to i
                        # c_49_j is the color assigned to j
                        # Get them from S
                        c_49_i = -1
                        c_49_j = -1
                        for c in range(4):
                            if i in S[c]: c_49_i = c
                            if j in S[c]: c_49_j = c
                            
                        if c_ij == c_49_i and c_ij == c_49_j:
                            valid_new = False
                            break
                    if not valid_new: break
                    
                if valid_new:
                    return {'n': 50, 'colors': new_colors}
                    
        return {'n': best_n, 'colors': best_colors}

    except Exception as e:
        return fallback_construct()
