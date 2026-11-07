"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 15}

def _degrees_from_mask(mask, n):
    """Compute degree of each vertex from an edge bitmask."""
    deg = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if mask & (1 << (i*n + j)):
                deg[i] += 1
                deg[j] += 1
    return deg

def _is_valid_mask(mask, n):
    """Check if the graph represented by mask satisfies the constraint."""
    deg = _degrees_from_mask(mask, n)
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if mask & (1 << (i*n + j)):
                adj[i].add(j)
                adj[j].add(i)
    
    # Group vertices by degree
    deg_to_verts = {}
    for i in range(n):
        deg_to_verts.setdefault(deg[i], []).append(i)
    
    # For each pair of vertices with equal degree, check if a length-3 path exists
    for d, verts in deg_to_verts.items():
        if len(verts) < 2:
            continue
        for idx_s in range(len(verts)):
            for idx_t in range(idx_s+1, len(verts)):
                s = verts[idx_s]
                t = verts[idx_t]
                # Check if path s-u-v-t exists with all distinct vertices
                for u in adj[s]:
                    if u == t:
                        continue
                    for v in adj[u]:
                        if v == s or v == t:
                            continue
                        if t in adj[v]:
                            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin before 60s limit
    
    n = P['n']
    
    # Start with K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    
    mask = 0
    for u in A:
        for v in B:
            mask |= (1 << (u*n + v))
    
    best_mask = mask
    best_count = mask.bit_count()
    
    # Verify initial
    if not _is_valid_mask(mask, n):
        # Fallback
        mask = 0
        for v in range(1, n):
            mask |= (1 << (0*n + v))
        best_mask = mask
        best_count = mask.bit_count()
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if mask & (1 << (i*n + j)):
                    edges.append([i, j])
        return edges
    
    # Local search: try adding edges
    while time.time() < deadline:
        improved = False
        
        # Try adding each missing edge
        for i in range(n):
            for j in range(i+1, n):
                if not (mask & (1 << (i*n + j))):
                    new_mask = mask | (1 << (i*n + j))
                    if _is_valid_mask(new_mask, n):
                        mask = new_mask
                        best_mask = mask
                        best_count = mask.bit_count()
                        improved = True
        
        # Try removing one edge and adding two different ones (2-opt style)
        if not improved and time.time() < deadline - 10:
            # Try swapping: remove edge e1, add edges e2, e3
            edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if mask & (1 << (i*n + j)):
                        edge_list.append((i, j))
            
            if len(edge_list) > 0:
                # Try a few random swaps
                for _ in range(100):
                    if time.time() >= deadline - 5:
                        break
                    e1 = random.choice(edge_list)
                    # Find two non-edges to add
                    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                                 if not (mask & (1 << (i*n + j)))]
                    if len(non_edges) >= 2:
                        e2, e3 = random.sample(non_edges, 2)
                        new_mask = mask
                        new_mask &= ~(1 << (e1[0]*n + e1[1]))
                        new_mask |= (1 << (e2[0]*n + e2[1]))
                        new_mask |= (1 << (e3[0]*n + e3[1]))
                        if new_mask.bit_count() > best_count and _is_valid_mask(new_mask, n):
                            mask = new_mask
                            best_mask = mask
                            best_count = mask.bit_count()
                            improved = True
                            break
        
        if not improved:
            # Perturbation: randomly flip some edges
            if time.time() >= deadline - 5:
                break
            for _ in range(50):
                if time.time() >= deadline - 3:
                    break
                perm_mask = mask
                for _ in range(random.randint(3, 8)):
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i != j:
                        if i > j:
                            i, j = j, i
                        perm_mask ^= (1 << (i*n + j))
                if _is_valid_mask(perm_mask, n) and perm_mask.bit_count() > best_count:
                    mask = perm_mask
                    best_mask = mask
                    best_count = mask.bit_count()
                    break
    
    # Convert best_mask to edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_mask & (1 << (i*n + j)):
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
