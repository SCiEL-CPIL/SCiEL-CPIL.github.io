import time
import random

def construct():
    start = time.time()
    deadline = start + 110
    
    n = 20
    
    # GF(4) arithmetic
    # Elements: 0, 1, 2, 3
    # 2 is omega, 3 is omega^2
    add_table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
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
        
    def gf_dot(p, q):
        return gf_add(gf_add(gf_mul(p[0], q[0]), gf_mul(p[1], q[1])), gf_mul(p[2], q[2]))
        
    # Generate 21 points of PG(2,4)
    points = []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize
                if a != 0:
                    # a is 1, 2, or 3. We want first non-zero to be 1.
                    # If a==1, keep.
                    # If a==2, multiply by 2 (inverse of 2 is 2).
                    # If a==3, multiply by 3 (inverse of 3 is 3).
                    inv_a = 1 if a == 1 else (2 if a == 2 else 3)
                    p = [gf_mul(a, inv_a), gf_mul(b, inv_a), gf_mul(c, inv_a)]
                elif b != 0:
                    inv_b = 1 if b == 1 else (2 if b == 2 else 3)
                    p = [0, gf_mul(b, inv_b), gf_mul(c, inv_b)]
                else: # c != 0
                    p = [0, 0, 1] # c must be 1, 2, or 3. If c!=1, scale to 1.
                    # Actually if a=0, b=0, c!=0.
                    # We want c=1.
                    # p = [0,0,1] is the only normalized form.
                    # But we iterate c in 1,2,3.
                    # So we will generate [0,0,1] three times.
                    # We need to avoid duplicates.
                    pass
                
                # Better way: just generate all non-zero vectors, normalize, and use a set.
                pass

    # Let's use a set to avoid duplicates
    point_set = set()
    for a in range(4):
        for b in range(4):
            for c in range(4):
                if a == 0 and b == 0 and c == 0:
                    continue
                if a != 0:
                    inv_a = 1 if a == 1 else (2 if a == 2 else 3)
                    p = (gf_mul(a, inv_a), gf_mul(b, inv_a), gf_mul(c, inv_a))
                elif b != 0:
                    inv_b = 1 if b == 1 else (2 if b == 2 else 3)
                    p = (0, gf_mul(b, inv_b), gf_mul(c, inv_b))
                else:
                    p = (0, 0, 1)
                point_set.add(p)
                
    points = list(point_set)
    # Should be 21 points
    assert len(points) == 21, f"Expected 21 points, got {len(points)}"
    
    # Map points to indices
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Build adjacency for 21-vertex graph
    adj_21 = [0] * 21
    absolute_points = []
    
    for i in range(21):
        pi = points[i]
        # Check if absolute
        if gf_dot(pi, pi) == 0:
            absolute_points.append(i)
            
        for j in range(i + 1, 21):
            pj = points[j]
            if gf_dot(pi, pj) == 0:
                adj_21[i] |= (1 << j)
                adj_21[j] |= (1 << i)
                
    # We have 5 absolute points
    # Remove one absolute point to get 20 vertices
    # Let's remove absolute_points[0]
    remove_idx = absolute_points[0]
    
    # Create mapping from old indices to new indices
    new_indices = [i for i in range(21) if i != remove_idx]
    old_to_new = {old: new for new, old in enumerate(new_indices)}
    
    # Build 20-vertex graph
    adj = [0] * 20
    for old_u in new_indices:
        new_u = old_to_new[old_u]
        mask = adj_21[old_u]
        # Filter out removed vertex
        mask &= ~(1 << remove_idx)
        # Remap indices
        new_mask = 0
        m = mask
        while m:
            lsb = m & (-m)
            m ^= lsb
            old_v = lsb.bit_length() - 1
            if old_v in old_to_new:
                new_v = old_to_new[old_v]
                new_mask |= (1 << new_v)
        adj[new_u] = new_mask
        
    # Count edges
    def popcount(x):
        cnt = 0
        while x:
            x &= x - 1
            cnt += 1
        return cnt

    edge_count = sum(popcount(adj[i]) for i in range(20)) // 2
    
    # Local search to try to add more edges
    def try_add_edge(adj, u, v):
        N_u = adj[u]
        N_v = adj[v]
        cu = N_u
        while cu:
            lsb_c = cu & (-cu)
            cu ^= lsb_c
            c = lsb_c.bit_length() - 1
            cd = N_u & (~lsb_c)
            while cd:
                lsb_d = cd & (-cd)
                cd ^= lsb_d
                d = lsb_d.bit_length() - 1
                candidates = N_v & adj[c] & adj[d]
                candidates &= ~(1 << u)
                candidates &= ~(1 << v)
                if popcount(candidates) >= 2:
                    return False
        return True

    # Try to add edges
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Iterate over all non-edges
        for u in range(20):
            for v in range(u + 1, 20):
                if not (adj[u] & (1 << v)):
                    if try_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edge_count += 1
                        improved = True
    
    # Convert to edge list
    edges_list = []
    for u in range(20):
        v_mask = adj[u]
        while v_mask:
            lsb = v_mask & (-v_mask)
            v_mask ^= lsb
            v = lsb.bit_length() - 1
            if u < v:
                edges_list.append([u, v])
                
    return edges_list
