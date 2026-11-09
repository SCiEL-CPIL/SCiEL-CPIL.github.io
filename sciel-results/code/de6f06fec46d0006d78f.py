import random
import itertools

def construct():
    # Approach: Expand the n=42 incidence graph of PG(2,4) to n=46.
    # Base graph: n=42, min degree 5.
    # Target: n=46, min degree 7.
    # Need to add 4 vertices and enough edges to raise min degree to 7.
    
    def get_pg2_4_graph():
        # Returns edges of the incidence graph of PG(2,4)
        # Vertices 0..20 are points, 21..41 are lines.
        # We construct the incidence structure.
        
        # GF(4) representation
        # 0: 0
        # 1: 1
        # 2: alpha
        # 3: alpha^2 = alpha + 1
        # Add: (a+b)%4 is not correct. Use lookup table.
        # Let's use integers 0,1,2,3.
        # 0+0=0, 0+1=1, 0+2=2, 0+3=3
        # 1+1=0, 1+2=3, 1+3=2
        # 2+2=1, 2+3=0
        # 3+3=2
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 1, 0],
            [3, 2, 0, 1]
        ]
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        
        def gf_add(a, b):
            return add_table[a][b]
        
        def gf_mul(a, b):
            return mul_table[a][b]
        
        # Generate points
        points = []
        seen = set()
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: make first non-zero element 1
                    # Or just collect unique projective classes
                    vec = (x, y, z)
                    # Find scalar multiple that is canonical?
                    # Just add to list and deduplicate by scaling.
                    points.append(vec)
        
        # Deduplicate points
        unique_points = []
        for p in points:
            x, y, z = p
            # Find a representative
            # If x != 0, scale so x=1
            # If x==0 and y != 0, scale so y=1
            # If x==0 and y==0, z != 0, scale so z=1
            
            if x != 0:
                # find inv x
                inv_x = 1
                for i in range(1, 4):
                    if gf_mul(i, x) == 1:
                        inv_x = i
                        break
                nx = gf_mul(inv_x, x)
                ny = gf_mul(inv_x, y)
                nz = gf_mul(inv_x, z)
                rep = (nx, ny, nz)
            elif y != 0:
                inv_y = 1
                for i in range(1, 4):
                    if gf_mul(i, y) == 1:
                        inv_y = i
                        break
                nx = gf_mul(inv_y, x)
                ny = gf_mul(inv_y, y)
                nz = gf_mul(inv_y, z)
                rep = (nx, ny, nz)
            else:
                # z != 0
                inv_z = 1
                for i in range(1, 4):
                    if gf_mul(i, z) == 1:
                        inv_z = i
                        break
                nx = gf_mul(inv_z, x)
                ny = gf_mul(inv_z, y)
                nz = gf_mul(inv_z, z)
                rep = (nx, ny, nz)
            
            if rep not in seen:
                seen.add(rep)
                unique_points.append(rep)
        
        # There should be 21 points
        # Map points to indices 0..20
        point_indices = {p: i for i, p in enumerate(unique_points)}
        
        # Generate lines
        # A line is a set of 3 collinear points.
        # In PG(2,q), lines can be represented by equations ax + by + cz = 0.
        # Coefficients (a,b,c) up to scale.
        lines = []
        seen_lines = set()
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    if a == 0 and b == 0 and c == 0:
                        continue
                    # Normalize
                    if a != 0:
                        inv_a = 1
                        for i in range(1, 4):
                            if gf_mul(i, a) == 1:
                                inv_a = i
                                break
                        na = gf_mul(inv_a, a)
                        nb = gf_mul(inv_a, b)
                        nc = gf_mul(inv_a, c)
                        rep = (na, nb, nc)
                    elif b != 0:
                        inv_b = 1
                        for i in range(1, 4):
                            if gf_mul(i, b) == 1:
                                inv_b = i
                                break
                        na = gf_mul(inv_b, a)
                        nb = gf_mul(inv_b, b)
                        nc = gf_mul(inv_b, c)
                        rep = (na, nb, nc)
                    else:
                        inv_c = 1
                        for i in range(1, 4):
                            if gf_mul(i, c) == 1:
                                inv_c = i
                                break
                        na = gf_mul(inv_c, a)
                        nb = gf_mul(inv_c, b)
                        nc = gf_mul(inv_c, c)
                        rep = (na, nb, nc)
                    
                    if rep not in seen_lines:
                        seen_lines.add(rep)
                        lines.append(rep)
        
        # Map lines to indices 21..41
        line_indices = {l: 21 + i for i, l in enumerate(lines)}
        
        edges = []
        # Check incidence
        for p_idx, p in enumerate(unique_points):
            x, y, z = p
            for l_idx, l in enumerate(lines):
                a, b, c = l
                val = gf_add(gf_add(gf_mul(a, x), gf_mul(b, y)), gf_mul(c, z))
                if val == 0:
                    edges.append([p_idx, 21 + l_idx])
        
        return edges

    # 1. Get base graph n=42
    base_edges = get_pg2_4_graph()
    n_base = 42
    n_target = 46
    target_deg = n_target - 39 # 7
    
    # Build adjacency sets for base graph
    G = {i: set() for i in range(n_target)}
    for u, v in base_edges:
        G[u].add(v)
        G[v].add(u)
    
    # Current degrees
    # Base vertices (0-41) have degree 5.
    # New vertices (42-45) have degree 0.
    
    # We need to add edges to reach min degree 7.
    # Deficit for base vertices: 7 - 5 = 2.
    # Deficit for new vertices: 7 - 0 = 7.
    
    # Possible new edges:
    # 1. Between new vertices (42-45)
    # 2. Between new and base (42-45, 0-41)
    
    # We will use a greedy approach to add edges.
    # To avoid C4, we check if adding (u,v) creates a C4.
    # C4 check: exists path u-a-b-v in G.
    
    def has_c4_after_add(G, u, v, n):
        N_u = G[u]
        N_v = G[v]
        for a in N_u:
            if a == v: continue
            for b in N_v:
                if b == u: continue
                if a == b: continue
                if b in G[a]:
                    return True
        return False

    # Candidate edges
    candidates = []
    # New-New
    for i in range(42, 46):
        for j in range(i+1, 46):
            candidates.append((i, j))
    # New-Base
    for i in range(42, 46):
        for j in range(42):
            candidates.append((i, j))
    
    random.shuffle(candidates)
    
    # Try multiple random orders
    for attempt in range(1000):
        G = {i: set() for i in range(n_target)}
        for u, v in base_edges:
            G[u].add(v)
            G[v].add(u)
        
        candidates = []
        for i in range(42, 46):
            for j in range(i+1, 46):
                candidates.append((i, j))
        for i in range(42, 46):
            for j in range(42):
                candidates.append((i, j))
        
        random.shuffle(candidates)
        
        edges = list(base_edges)
        
        # Greedy add
        for u, v in candidates:
            if len(G[u]) >= target_deg and len(G[v]) >= target_deg:
                continue
            # Check if adding helps reduce deficit
            # We just add if no C4
            if not has_c4_after_add(G, u, v, n_target):
                G[u].add(v)
                G[v].add(u)
                edges.append([u, v])
        
        # Check min degree
        min_deg = min(len(G[i]) for i in range(n_target))
        if min_deg >= target_deg:
            return {'n': n_target, 'edges': edges}
            
    # Fallback to n=42
    return {'n': 42, 'edges': base_edges}
