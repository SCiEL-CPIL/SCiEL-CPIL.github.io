import time
import random

def construct():
    n = 18
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Construct Paley(17) on vertices 0..16
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    adj = [0] * 17
    for u in range(17):
        for v in range(u + 1, 17):
            diff = (v - u) % 17
            if diff in residues:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # 2. Precompute is_indep for all masks
    is_indep = [False] * (1 << 17)
    is_indep[0] = True
    for mask in range(1, 1 << 17):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True

    # Precompute triangles
    triangles = []
    for u in range(17):
        for v in range(u + 1, 17):
            if adj[u] & (1 << v):
                for w in range(v + 1, 17):
                    if (adj[u] & (1 << w)) and (adj[v] & (1 << w)):
                        triangles.append((1 << u) | (1 << v) | (1 << w))
                        
    # Precompute TF sets of size 8
    tf_sets_8_masks = []
    for mask in range(1 << 17):
        if bin(mask).count('1') != 8:
            continue
        is_tf = True
        for tri in triangles:
            if (mask & tri) == tri:
                is_tf = False
                break
        if is_tf:
            tf_sets_8_masks.append(mask)
            
    # Helper to check if mask is TF
    def is_mask_tf(mask):
        for tri in triangles:
            if (mask & tri) == tri:
                return False
        return True
        
    # Helper to check validity
    def is_valid_a(a_mask):
        for t_mask in tf_sets_8_masks:
            inter = t_mask & a_mask
            if is_indep[inter]:
                return False
        return True

    random.seed(42)
    found_a = None
    while time.time() < deadline - 5.0:
        # Generate random TF set
        a_mask = 0
        verts = list(range(17))
        random.shuffle(verts)
        for v in verts:
            new_mask = a_mask | (1 << v)
            if is_mask_tf(new_mask):
                a_mask = new_mask
        
        if is_valid_a(a_mask):
            found_a = a_mask
            break
            
    edges = []
    for u in range(17):
        for v in range(u + 1, 17):
            if adj[u] & (1 << v):
                edges.append([u, v])
                
    if found_a is not None:
        for v in range(17):
            if found_a & (1 << v):
                edges.append([17, v])
    # If found_a is None, 17 is isolated (no edges added)
    
    return edges
