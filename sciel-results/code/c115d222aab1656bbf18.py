import time
import random

def _q52_graph():
    """
    Generate the point graph of Q(5,2).
    Vertices are non-zero vectors in GF(2)^6 with Q(v)=0.
    Q(x) = x0x1 + x2x3 + x4x5.
    Adjacency: B(u,v) = 0.
    """
    pts = []
    for i in range(1, 64):
        # Check Q(i) = 0
        x0, x1 = i & 1, (i >> 1) & 1
        x2, x3 = (i >> 2) & 1, (i >> 3) & 1
        x4, x5 = (i >> 4) & 1, (i >> 5) & 1
        q_val = (x0 * x1 + x2 * x3 + x4 * x5) % 2
        if q_val == 0:
            pts.append(i)
    
    n = len(pts)
    edges = []
    
    for i in range(n):
        u = pts[i]
        for j in range(i + 1, n):
            v = pts[j]
            # B(u, v)
            u0, u1 = u & 1, (u >> 1) & 1
            u2, u3 = (u >> 2) & 1, (u >> 3) & 1
            u4, u5 = (u >> 4) & 1, (u >> 5) & 1
            
            v0, v1 = v & 1, (v >> 1) & 1
            v2, v3 = (v >> 2) & 1, (v >> 3) & 1
            v4, v5 = (v >> 4) & 1, (v >> 5) & 1
            
            b_val = (u0 * v1 + u1 * v0 + u2 * v3 + u3 * v2 + u4 * v5 + u5 * v4) % 2
            if b_val == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}

def _has_indep_k(adj, n, k, deadline):
    """
    Check if an independent set of size k exists.
    """
    def backtrack(remaining, current_size):
        if time.time() > deadline:
            return False
        if current_size >= k:
            return True
        if not remaining:
            return False
        if current_size + remaining.bit_count() < k:
            return False
        
        # Pick vertex with min degree in remaining
        min_v = -1
        min_deg = 100
        temp = remaining
        while temp:
            v = (temp & -temp).bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            temp &= temp - 1
            
        if min_v == -1:
            return False
            
        # Include min_v
        new_remaining = remaining & ~(adj[min_v] | (1 << min_v))
        if backtrack(new_remaining, current_size + 1):
            return True
            
        # Exclude min_v
        remaining &= ~(1 << min_v)
        if backtrack(remaining, current_size):
            return True
            
        return False

    return backtrack((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # Generate Q(5,2)
    g = _q52_graph()
    n_orig = g['n']
    edges_orig = g['edges']
    
    # Build adjacency bitmask
    adj_orig = [0] * n_orig
    for u, v in edges_orig:
        adj_orig[u] |= 1 << v
        adj_orig[v] |= 1 << u
        
    # Try to prune to n=40, then 39, etc.
    # We want to remove 'rem' vertices such that alpha < 11.
    
    best_n = 0
    best_edges = []
    
    # Fallback: 10 triangles (n=30)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    best_n = 30
    best_edges = fallback_edges
    
    # Try removing 11 vertices (n=40)
    # Then 12 (n=39)
    
    for rem in [11, 12, 13, 14, 15]:
        if time.time() > deadline - 10:
            break
            
        target_n = n_orig - rem
        if target_n <= best_n:
            break
            
        # Randomly select vertices to remove
        found_valid = False
        for _ in range(100): # 100 attempts per rem count
            if time.time() > deadline - 5:
                break
                
            remove_set = set(random.sample(range(n_orig), rem))
            keep_set = set(range(n_orig)) - remove_set
            keep_list = list(keep_set)
            new_n = len(keep_list)
            
            # Map old indices to new indices
            old_to_new = {old: new for new, old in enumerate(keep_list)}
            
            # Build new adjacency
            new_adj = [0] * new_n
            for old_u in keep_list:
                new_u = old_to_new[old_u]
                for old_v in keep_list:
                    if old_u == old_v: continue
                    if adj_orig[old_u] & (1 << old_v):
                        new_adj[new_u] |= 1 << old_to_new[old_v]
                        
            # Check alpha < 11
            if not _has_indep_k(new_adj, new_n, 11, deadline):
                # Valid
                # Build edge list
                new_edges = []
                for i in range(new_n):
                    for j in range(i + 1, new_n):
                        if new_adj[i] & (1 << j):
                            new_edges.append([i, j])
                            
                if new_n > best_n:
                    best_n = new_n
                    best_edges = new_edges
                    found_valid = True
                break # Found one for this rem, try smaller rem? No, we want max n, so smaller rem is better.
                # But we iterate rem from small to large (11, 12...), so we find max n first.
                
        if found_valid:
            break # Found a valid graph for this rem, since we start with min rem, this is max n.
            
    return {'n': best_n, 'edges': best_edges}
