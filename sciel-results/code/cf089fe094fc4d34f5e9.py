import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    def popcount(x):
        # Python 3.10+ has int.bit_count()
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def is_safe(adj, u, v):
        """
        Check if adding edge (u, v) creates a C4.
        adj is a list of integers (bitmasks).
        C4 exists if there is a path u-a-b-v.
        This means exists a in N(u), b in N(v) such that a~b.
        Equivalent to: exists a in N(u) such that N(a) intersects N(v).
        Or: exists b in N(v) such that N(b) intersects N(u).
        """
        mu = adj[u]
        mv = adj[v]
        
        # Iterate over the smaller neighbor set
        if popcount(mu) <= popcount(mv):
            # Iterate through set bits in mu
            mask = mu
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                a = lsb.bit_length() - 1
                # Check if a is connected to any neighbor of v
                if adj[a] & mv:
                    return False
                mask ^= lsb
        else:
            # Iterate through set bits in mv
            mask = mv
            while mask:
                lsb = mask & -mask
                b = lsb.bit_length() - 1
                # Check if b is connected to any neighbor of u
                if adj[b] & mu:
                    return False
                mask ^= lsb
        return True

    def try_construct(n, min_deg_req, max_restarts=5000):
        """
        Try to construct a C4-free graph on n vertices with min degree >= min_deg_req.
        Returns list of edges if successful, None otherwise.
        """
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            adj = [0] * n
            deg = [0] * n
            
            # Greedy construction
            # We need to add edges until min(deg) >= min_deg_req
            # To avoid getting stuck, we prioritize low-degree vertices.
            # We also add randomness to break ties and explore different topologies.
            
            success = True
            # We need to ensure we don't loop forever. 
            # Max edges needed is roughly n * min_deg_req / 2.
            # But we might add "extra" edges if we aren't careful? 
            # No, we stop when min(deg) >= min_deg_req.
            # However, adding an edge increases two degrees.
            # We should stop as soon as the condition is met.
            
            while min(deg) < min_deg_req:
                if time.time() > deadline:
                    return None
                
                # Find vertex with minimum degree
                # To add randomness, we can pick randomly from the bottom 20% of degrees
                # or just the absolute minimum with random tie-breaking.
                # Let's pick the absolute minimum, but shuffle the order of checking 
                # or pick randomly from vertices with deg == min_deg.
                
                min_d = min(deg)
                candidates_u = [i for i in range(n) if deg[i] == min_d]
                u = random.choice(candidates_u)
                
                # Find safe v to connect to u
                # We prefer v with low degree to balance the graph.
                # But we must ensure v is not already connected to u.
                
                # Collect all safe candidates
                safe_vs = []
                for v in range(n):
                    if v == u:
                        continue
                    if (adj[u] >> v) & 1:
                        continue # already connected
                    
                    if is_safe(adj, u, v):
                        safe_vs.append(v)
                
                if not safe_vs:
                    success = False
                    break
                
                # Pick v from safe_vs
                # Strategy: Pick v with lowest degree. 
                # To avoid determinism, pick randomly from the lowest 10% of safe_vs degrees?
                # Or just the absolute lowest degree.
                # Let's sort safe_vs by degree and pick from the bottom few.
                safe_vs.sort(key=lambda x: deg[x])
                
                # Pick from the first max(1, len(safe_vs)//10) candidates
                k = max(1, len(safe_vs) // 10)
                v = random.choice(safe_vs[:k])
                
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
            
            if success and min(deg) >= min_deg_req:
                # Verify C4-free property (redundant if is_safe is correct, but good for safety)
                has_c4 = False
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            common = adj[i] & adj[j]
                            # If more than 1 common neighbor, C4 exists
                            if common & (common - 1):
                                has_c4 = True
                                break
                    if has_c4:
                        break
                
                if not has_c4:
                    # Convert to edge list
                    edges_list = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if (adj[i] >> j) & 1:
                                edges_list.append([i, j])
                    return edges_list
        
        return None

    # Try n=46 first, as it is the maximum possible (n=47 is impossible by Reiman bound)
    # n=46 requires min_deg = 7
    # n=45 requires min_deg = 6
    
    # Allocate time: 120s for n=46, remaining for n=45
    t_limit_46 = start_time + 120.0
    
    # Override deadline for the 46 attempt
    global_deadline = deadline
    deadline = t_limit_46
    
    edges = try_construct(46, 7, max_restarts=10000)
    
    # Restore deadline
    deadline = global_deadline
    
    if edges is not None:
        return {'n': 46, 'edges': edges}
        
    # Fallback to n=45
    edges = try_construct(45, 6, max_restarts=5000)
    if edges is not None:
        return {'n': 45, 'edges': edges}
        
    # Fallback to n=44
    edges = try_construct(44, 5, max_restarts=2000)
    if edges is not None:
        return {'n': 44, 'edges': edges}

    # Final fallback: n=40, min_deg=1 (always possible)
    # Simple path or star? Star K1,39 is not C4-free? 
    # Star K1,n-1 has no cycles. Min degree 1 (for leaves). 
    # But we need min degree >= 1. 
    # A star on 40 vertices has min degree 1 (leaves) and 39 (center). 
    # Is it C4-free? Yes, no cycles.
    # But wait, min degree must be >= n-39 = 1. 
    # Star works.
    edges = [[0, i] for i in range(1, 40)]
    return {'n': 40, 'edges': edges}
