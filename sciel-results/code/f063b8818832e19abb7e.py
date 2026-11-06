import time
import random

def _creates_c4(adj_int, n, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    A C4 is created if u and v have at least 2 common neighbors.
    adj_int: list of integers representing adjacency.
    u, v: vertices to connect. Assumes not already connected.
    """
    nu = adj_int[u]
    nv = adj_int[v]
    
    # Check common neighbors: if |N(u) ∩ N(v)| >= 2, adding (u,v) creates C4
    common = nu & nv
    if common & (common - 1):  # More than 1 bit set
        return True
        
    return False

def _find_c4(adj_int, n):
    """Find a C4 in the graph. Returns (u, v, w, x) forming a C4, or None."""
    for u in range(n):
        for v in range(u+1, n):
            common = adj_int[u] & adj_int[v]
            if common & (common - 1):  # >= 2 common neighbors
                # Get two common neighbors
                lsb1 = common & -common
                w = lsb1.bit_length() - 1
                common_rest = common ^ lsb1
                lsb2 = common_rest & -common_rest
                x = lsb2.bit_length() - 1
                return (u, v, w, x)
    return None

def _construct_c4_free(n, min_deg_req, time_limit):
    start_time = time.time()
    
    # Try multiple random initializations
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random graph
        p = random.uniform(0.5, 0.85)
        
        adj_int = [0] * n
        degrees = [0] * n
        
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj_int[u] |= (1 << v)
                    adj_int[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
        
        # Prune C4s
        changed = True
        prune_count = 0
        while changed:
            if time.time() - start_time > time_limit:
                return None
            changed = False
            
            # Find a C4
            c4 = _find_c4(adj_int, n)
            if c4 is None:
                break
                
            u, v, w, x = c4
            # The C4 is u-w-...-v-x-u? No, u and v share neighbors w and x.
            # So the C4 is u-w-v-x-u (if edges u-w, w-v, v-x, x-u exist)
            # Actually: u is connected to w and x, v is connected to w and x.
            # So C4: u-w-v-x-u
            # Edges: (u,w), (w,v), (v,x), (x,u)
            
            # Remove the edge incident to the highest degree vertex
            candidates = [
                (u, w, degrees[u]),
                (w, v, degrees[w]),
                (v, x, degrees[v]),
                (x, u, degrees[x])
            ]
            candidates.sort(key=lambda t: t[2], reverse=True)
            a, b, _ = candidates[0]
            
            # Remove edge (a, b)
            adj_int[a] &= ~(1 << b)
            adj_int[b] &= ~(1 << a)
            degrees[a] -= 1
            degrees[b] -= 1
            changed = True
            prune_count += 1
            
            # Limit pruning iterations to avoid infinite loops
            if prune_count > 10000:
                break
        
        # After pruning, check min degree
        min_deg = min(degrees)
        if min_deg >= min_deg_req:
            # Convert to edge list
            edges = []
            for u in range(n):
                temp = adj_int[u]
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    temp ^= lsb
            return {'n': n, 'edges': edges}
            
        # If min_deg is close, try to add edges to low-degree vertices
        if min_deg >= min_deg_req - 3:
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                min_idx = min(range(n), key=lambda i: degrees[i])
                if degrees[min_idx] >= min_deg_req:
                    break
                
                # Try all possible neighbors for the low-degree vertex
                added = False
                for v in range(n):
                    if v == min_idx or (adj_int[min_idx] >> v) & 1:
                        continue
                    if _creates_c4(adj_int, n, min_idx, v):
                        continue
                    adj_int[min_idx] |= (1 << v)
                    adj_int[v] |= (1 << min_idx)
                    degrees[min_idx] += 1
                    degrees[v] += 1
                    added = True
                    break
                
                if not added:
                    break
                    
            min_deg = min(degrees)
            if min_deg >= min_deg_req:
                edges = []
                for u in range(n):
                    temp = adj_int[u]
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        temp ^= lsb
                return {'n': n, 'edges': edges}
        
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 1:
            break
            
        min_deg_req = max(1, n - 39)
        t_limit = min(30.0, remaining - 1)
        
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result:
            return result
            
    return {'n': 39, 'edges': []}
