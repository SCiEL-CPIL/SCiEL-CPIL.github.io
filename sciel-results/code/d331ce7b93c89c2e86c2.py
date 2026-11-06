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

def _greedy_construct_c4_free(n, min_deg_req, time_limit):
    start_time = time.time()
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Try multiple random edge orderings
    for attempt in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        # Shuffle edge order
        random.shuffle(all_edges)
        
        adj_int = [0] * n
        degrees = [0] * n
        
        # Greedily add edges if they don't create a C4
        for u, v in all_edges:
            if time.time() - start_time > time_limit:
                break
            if not _creates_c4(adj_int, n, u, v):
                adj_int[u] |= (1 << v)
                adj_int[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
        
        # Check min degree
        min_deg = min(degrees) if degrees else 0
        if min_deg >= min_deg_req:
            # Convert to edge list
            res_edges = []
            for u in range(n):
                temp = adj_int[u]
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if u < v:
                        res_edges.append([u, v])
                    temp ^= lsb
            
            # Final Verification: Ensure no C4 exists
            is_valid = True
            for u in range(n):
                for v in range(u+1, n):
                    common = adj_int[u] & adj_int[v]
                    if common & (common - 1):
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                return {'n': n, 'edges': res_edges}
            else:
                # Should not happen with greedy, but safe check
                continue
                
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
        t_limit = min(20.0, remaining - 1)
        
        result = _greedy_construct_c4_free(n, min_deg_req, t_limit)
        if result:
            return result
            
    return {'n': 39, 'edges': []}
