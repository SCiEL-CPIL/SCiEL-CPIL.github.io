import random

def construct():
    n = 20
    
    # GF(4) implementation
    # Elements: 0, 1, 2, 3
    # 2 = alpha, 3 = alpha + 1
    # alpha^2 = alpha + 1 => 2*2 = 3
    # alpha^3 = 1
    # Multiplication table
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def gf_add(a, b):
        return a ^ b
        
    def gf_mul(a, b):
        return mul_table[a][b]
        
    def gf_neg(a):
        return a # Characteristic 2
        
    # Generate points of PG(2, 4)
    # Points are vectors (x, y, z) not all zero, modulo scaling by non-zero elements {1, 2, 3}
    points = []
    seen = set()
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonical form: make first non-zero element 1
                # Multiply by inverse of first non-zero element
                if x != 0:
                    # x is 1, 2, or 3
                    if x == 1:
                        inv = 1
                    elif x == 2:
                        inv = 2 # 2*2=3!=1, 2*3=1. Wait. 2*2=3, 2*3=1. So inv(2)=3.
                        # Let's check: 2*3 = 1. Yes.
                        inv = 3
                    elif x == 3:
                        inv = 2 # 3*2 = 1. Yes.
                        inv = 2
                    vx = gf_mul(x, inv)
                    vy = gf_mul(y, inv)
                    vz = gf_mul(z, inv)
                    p = (vx, vy, vz)
                    if p not in seen:
                        seen.add(p)
                        points.append(p)
                elif y != 0:
                    if y == 1:
                        inv = 1
                    elif y == 2:
                        inv = 3
                    elif y == 3:
                        inv = 2
                    vx = gf_mul(x, inv) # 0
                    vy = gf_mul(y, inv) # 1
                    vz = gf_mul(z, inv)
                    p = (vx, vy, vz)
                    if p not in seen:
                        seen.add(p)
                        points.append(p)
                elif z != 0:
                    if z == 1:
                        inv = 1
                    elif z == 2:
                        inv = 3
                    elif z == 3:
                        inv = 2
                    vx = gf_mul(x, inv) # 0
                    vy = gf_mul(y, inv) # 0
                    vz = gf_mul(z, inv) # 1
                    p = (vx, vy, vz)
                    if p not in seen:
                        seen.add(p)
                        points.append(p)
                        
    # We should have 21 points
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Compute adjacency for all 21 points
    # Adjacency: x0*y0 + x1*y1 + x2*y2 == 0
    adj_21 = [0] * 21
    for i in range(21):
        p1 = points[i]
        for j in range(21):
            if i == j:
                continue
            p2 = points[j]
            dot = gf_add(gf_add(gf_mul(p1[0], p2[0]), gf_mul(p1[1], p2[1])), gf_mul(p1[2], p2[2]))
            if dot == 0:
                adj_21[i] |= (1 << j)
                
    # Remove vertex 20
    # We keep vertices 0..19
    # We need to construct the graph on 0..19
    adj = [0] * n
    for i in range(n):
        # Copy bits from adj_21[i], but only for vertices 0..19
        mask = adj_21[i] & ((1 << n) - 1)
        adj[i] = mask
        
    # Now perform local search to add edges
    def is_valid_add(u, v):
        nu = adj[u]
        nv = adj[v]
        
        # Check pairs in N(u)
        mask = nu
        while mask:
            lsb_x = mask & -mask
            x = lsb_x.bit_length() - 1
            mask ^= lsb_x
            
            mask_y = mask
            while mask_y:
                lsb_y = mask_y & -mask_y
                y = lsb_y.bit_length() - 1
                mask_y ^= lsb_y
                
                common = adj[v] & adj[x] & adj[y]
                if common.bit_count() > 1:
                    return False
        
        # Check pairs in N(v)
        mask = nv
        while mask:
            lsb_x = mask & -mask
            x = lsb_x.bit_length() - 1
            mask ^= lsb_x
            
            mask_y = mask
            while mask_y:
                lsb_y = mask_y & -mask_y
                y = lsb_y.bit_length() - 1
                mask_y ^= lsb_y
                
                common = adj[u] & adj[x] & adj[y]
                if common.bit_count() > 1:
                    return False
                    
        return True

    # Try to add edges
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    if is_valid_add(u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        changed = True

    # Convert to list of edges
    res = []
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            if u < v:
                res.append([u, v])
                
    return res
