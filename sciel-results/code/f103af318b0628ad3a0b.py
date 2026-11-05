import time
import random
import numpy as np

def _popcount64(x):
    """Count set bits in a 64-bit integer."""
    return bin(x).count('1')

def _has_c4(mask, n):
    """Check if graph represented by bitmask adjacency has a C4.
    Returns (u, v, w) where w is a common neighbor of u,v if C4 found, else None."""
    for u in range(n):
        mu = mask[u]
        for v in range(u + 1, n):
            common = mu & mask[v]
            # Check if common has >= 2 bits set
            # Quick check: common & (common - 1) != 0 means at least 2 bits
            if common & (common - 1):
                # Find two distinct common neighbors
                # Extract lowest set bit
                b1 = common & (-common)
                common_rest = common ^ b1
                b2 = common_rest & (-common_rest)
                w1 = b1.bit_length() - 1
                w2 = b2.bit_length() - 1
                return (u, v, w1, w2)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_n = 0
    best_edges = []
    
    # Try n=47 first, then 46
    for target_n in [47, 46]:
        if time.time() > deadline:
            break
            
        min_deg_req = target_n - 39
        
        # Greedy edge addition with bitmask C4 checking
        # Strategy: start with empty graph, greedily add edges while maintaining C4-free
        # and targeting minimum degree >= min_deg_req
        
        found = False
        max_attempts = 200  # Many fast attempts
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use different random seeds / strategies per attempt
            random.seed(attempt * 7919 + 12345)
            
            adj_mask = [0] * target_n
            # Track degrees
            deg = [0] * target_n
            
            # Phase 1: Greedy add edges to reach minimum degree
            # Order vertices by current degree (process lowest first)
            # For each vertex, try to add edges to non-neighbors that don't create C4
            
            max_rounds = 100
            for rnd in range(max_rounds):
                if time.time() > deadline:
                    break
                
                # Check if all degrees are sufficient
                min_deg = min(deg)
                if min_deg >= min_deg_req:
                    break
                
                # Process vertices in order of increasing degree
                verts = list(range(target_n))
                verts.sort(key=lambda x: deg[x])
                
                for u in verts:
                    if deg[u] >= min_deg_req:
                        continue
                    if time.time() > deadline:
                        break
                    
                    # Try to add edges from u to non-neighbors
                    # Non-neighbors of u
                    non_neighbors = [v for v in range(target_n) if v != u and not (adj_mask[u] >> v) & 1]
                    random.shuffle(non_neighbors)
                    
                    added = 0
                    for v in non_neighbors:
                        if deg[u] >= min_deg_req:
                            break
                        if time.time() > deadline:
                            break
                        
                        # Check if adding (u,v) creates a C4
                        # C4 exists if u and v have >= 2 common neighbors
                        common = adj_mask[u] & adj_mask[v]
                        if common & (common - 1):
                            # Would create C4, skip
                            continue
                        
                        # Safe to add edge (u, v)
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                        added += 1
            
            # Phase 2: Verify C4-free and min degree
            min_deg = min(deg)
            if min_deg < min_deg_req:
                continue
            
            # Final C4 check
            c4_info = _has_c4(adj_mask, target_n)
            if c4_info is not None:
                continue
            
            # Success!
            found = True
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append([i, j])
            
            if target_n > best_n:
                best_n = target_n
                best_edges = edges
            if target_n == 47:
                return {'n': best_n, 'edges': best_edges}
        
        # If greedy didn't work, try random init + edge removal
        if not found:
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                random.seed(attempt * 31337 + 42)
                
                # Start with random dense graph
                p = min_deg_req / target_n * 1.3
                adj_mask = [0] * target_n
                deg = [0] * target_n
                
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if random.random() < p:
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
                            deg[u] += 1
                            deg[v] += 1
                
                # Remove edges to break C4s while maintaining min degree
                max_iter = 500
                for _ in range(max_iter):
                    if time.time() > deadline:
                        break
                    
                    c4_info = _has_c4(adj_mask, target_n)
                    if c4_info is None:
                        # No C4, check min degree
                        min_deg = min(deg)
                        if min_deg >= min_deg_req:
                            found = True
                            edges = []
                            for i in range(target_n):
                                for j in range(i + 1, target_n):
                                    if (adj_mask[i] >> j) & 1:
                                        edges.append([i, j])
                            if target_n > best_n:
                                best_n = target_n
                                best_edges = edges
                            if target_n == 47:
                                return {'n': best_n, 'edges': best_edges}
                        break
                    
                    u, v, w1, w2 = c4_info
                    # Remove one edge from the C4: (u,w1), (w1,v), (v,w2), (w2,u)
                    # Try each edge removal, pick one that doesn't violate min degree
                    candidates = [(u, w1), (w1, v), (v, w2), (w2, u)]
                    random.shuffle(candidates)
                    
                    removed = False
                    for a, b in candidates:
                        if deg[a] > min_deg_req and deg[b] > min_deg_req:
                            # Safe to remove
                            adj_mask[a] &= ~(1 << b)
                            adj_mask[b] &= ~(1 << a)
                            deg[a] -= 1
                            deg[b] -= 1
                            removed = True
                            break
                    
                    if not removed:
                        # Can't remove any edge without violating min degree
                        break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: empty graph on 39 vertices
    return {'n': 39, 'edges': []}
