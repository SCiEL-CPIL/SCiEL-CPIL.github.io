"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute edge indices
    edge_list = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u+1, n):
            edge_to_idx[(u, v)] = len(edge_list)
            edge_list.append((u, v))
    E = len(edge_list)  # 105
    
    # Precompute numpy arrays for fast edge extraction
    edge_u = np.array([e[0] for e in edge_list], dtype=np.int32)
    edge_v = np.array([e[1] for e in edge_list], dtype=np.int32)
    
    # Precompute for each edge, the other edges that share vertex u or v
    # edge_neighbors[i] = list of edge indices j != i where edge j shares a vertex with edge i
    edge_neighbors = [[] for _ in range(E)]
    for i in range(E):
        u_i, v_i = edge_list[i]
        for j in range(E):
            if j == i:
                continue
            u_j, v_j = edge_list[j]
            if u_j == u_i or v_j == u_i or u_j == v_i or v_j == v_i:
                edge_neighbors[i].append(j)
    
    def get_adj_deg(mask):
        """Build adjacency list and degree array from mask."""
        adj = [set() for _ in range(n)]
        deg = [0] * n
        # Use numpy for fast extraction
        mask_bits = np.array([(mask >> i) & 1 for i in range(E)], dtype=np.bool_)
        present_u = edge_u[mask_bits]
        present_v = edge_v[mask_bits]
        for k in range(len(present_u)):
            u = int(present_u[k])
            v = int(present_v[k])
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def is_valid_mask(mask):
        """Check if the given mask represents a valid graph."""
        adj, deg = get_adj_deg(mask)
        # For each present edge (u,v), check all paths s-u-v-t
        for i in range(E):
            if not (mask & (1 << i)):
                continue
            u, v = edge_list[i]
            ns = adj[u] - {v}
            nt = adj[v] - {u}
            if not ns or not nt:
                continue
            # Group neighbors by degree for faster comparison
            deg_ns = {s: deg[s] for s in ns}
            for t in nt:
                dt = deg[t]
                # Check if any s in ns has deg[s] == dt and s != t
                for s in ns:
                    if s != t and deg[s] == dt:
                        return False
        return True
    
    # Build initial graph: K_{7,8}
    A = set(range(7))
    B = set(range(7, 15))
    
    init_mask = 0
    for a in A:
        for b in B:
            init_mask |= (1 << edge_to_idx[(a, b)])
    
    # Check if K_{7,8} is valid
    if is_valid_mask(init_mask):
        best_mask = init_mask
        best_count = bin(init_mask).count('1')
    else:
        # Find best valid subset greedily
        best_mask = 0
        best_count = 0
        for i in range(E):
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
    
    # Try adding any edge (multiple passes)
    for _ in range(15):
        if time.time() > deadline - 5:
            break
        changed = False
        for i in range(E):
            if best_mask & (1 << i):
                continue
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
                changed = True
        if not changed:
            break
    
    # Local search with swaps
    rng = random.Random(42)
    for iteration in range(2000):
        if time.time() > deadline - 3:
            break
        
        present = [i for i in range(E) if best_mask & (1 << i)]
        absent = [i for i in range(E) if not (best_mask & (1 << i))]
        
        if not present or not absent:
            break
        
        improved = False
        
        # Try single add
        for i in absent:
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-2 swap
        if len(absent) >= 2:
            attempts = min(500, max(100, len(absent) * 2))
            for _ in range(attempts):
                if time.time() > deadline - 3:
                    break
                ri = rng.choice(present)
                a1 = rng.choice(absent)
                a2 = rng.choice(absent)
                if a1 == a2:
                    continue
                new_mask = (best_mask & ~(1 << ri)) | (1 << a1) | (1 << a2)
                if is_valid_mask(new_mask):
                    best_mask = new_mask
                    best_count += 1
                    improved = True
                    break
        
        if not improved and iteration > 200:
            # Perturb and re-greedy
            if iteration % 30 == 0:
                new_mask = best_mask
                k = min(4, best_count)
                for _ in range(k):
                    ri = rng.choice(present)
                    new_mask &= ~(1 << ri)
                # Re-add greedily
                for i in range(E):
                    if new_mask & (1 << i):
                        continue
                    try_mask = new_mask | (1 << i)
                    if is_valid_mask(try_mask):
                        new_mask = try_mask
                new_count = bin(new_mask).count('1')
                if new_count > best_count:
                    best_mask = new_mask
                    best_count = new_count
    
    # Final greedy pass
    for _ in range(5):
        if time.time() > deadline - 1:
            break
        changed = False
        for i in range(E):
            if best_mask & (1 << i):
                continue
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
                changed = True
        if not changed:
            break
    
    # Convert result
    result = []
    for i in range(E):
        if best_mask & (1 << i):
            u, v = edge_list[i]
            result.append([int(u), int(v)])
    result.sort()
    return result
# EVOLVE-BLOCK-END
