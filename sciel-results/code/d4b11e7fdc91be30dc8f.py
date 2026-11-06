import time
import random
import itertools

# EVOLVE-BLOCK-START

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # at least two bits set
                return True
    return False

def _max_independent_set_size_bitmask(n, adj, target=11):
    """
    Check if there exists an independent set of size >= target.
    Returns the size of the maximum independent set found (capped at target).
    Uses branch and bound with better vertex ordering.
    """
    # Precompute vertex order by degree (descending) for better pruning
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    order_map = {v: i for i, v in enumerate(order)}
    
    # Remap adjacencies to new ordering
    new_adj = [0] * n
    for i in range(n):
        old_v = order[i]
        mask = adj[old_v]
        new_mask = 0
        temp = mask
        while temp:
            lsb = temp & -temp
            old_u = lsb.bit_length() - 1
            new_u = order_map[old_u]
            new_mask |= (1 << new_u)
            temp ^= lsb
        new_adj[i] = new_mask
    
    def max_is(vertices_mask, remaining, best_so_far):
        if remaining <= 0:
            return 0
        if not vertices_mask:
            return 0
        
        # Upper bound: count vertices
        count = 0
        temp = vertices_mask
        while temp:
            temp &= temp - 1
            count += 1
        if count < remaining:
            return count
        if count + best_so_far < remaining:
            return count
        
        # Pick vertex with highest degree in remaining subgraph
        # Use first set bit for speed, but try a few candidates
        lsb = vertices_mask & -vertices_mask
        v = lsb.bit_length() - 1
        
        # Option 1: include v
        new_mask = vertices_mask & ~(1 << v) & ~new_adj[v]
        include_val = 1 + max_is(new_mask, remaining - 1, best_so_far)
        
        # Option 2: exclude v
        exclude_val = max_is(vertices_mask ^ (1 << v), remaining, best_so_far)
        
        return max(include_val, exclude_val)
    
    all_vertices = (1 << n) - 1
    return max_is(all_vertices, target, 0)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    if n < 1:
        return False
    adj = [0] * n
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if adj[u] & (1 << v):
            return False # Duplicate edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4_bitmask(n, adj):
        return False
    
    if _max_independent_set_size_bitmask(n, adj, 11) >= 11:
        return False
    
    return True

def _build_polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5).
    n = 31.
    """
    q = 5
    # Points of PG(2, q)
    points = []
    # Affine points: (1, x, y)
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    # Points at infinity: (0, 1, x)
    for x in range(q):
        points.append((0, 1, x))
    # Point at infinity: (0, 0, 1)
    points.append((0, 0, 1))
    
    n = len(points)
    adj = [0] * n
    edges = []
    
    # Polarity: u . v = 0 mod q
    # Using standard dot product
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
                
    return n, adj, edges

def _construct_large_graph():
    """
    Try to construct a graph with n > 31.
    Starts with PG(2,5) polarity graph and tries to extend.
    """
    deadline = time.time() + 110
    
    # Base graph
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    
    # Check if base is valid (it should be)
    if not _verify_graph(n_base, edges_base):
        # Fallback to a known valid small graph if polarity fails (should not happen)
        # Disjoint triangles
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    best_result = {'n': n_base, 'edges': edges_base}
    best_n = n_base
    
    # Try to extend to n = 39, 38, ..., 32
    # We will try random extensions from the base graph
    # To save time, we limit the number of restarts
    
    for target_n in range(39, n_base, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random extensions
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 1000 + target_n)
            
            adj = adj_base.copy()
            edges = edges_base.copy()
            n = n_base
            
            # Add (target_n - n_base) vertices
            # We need to assign neighbors for each new vertex
            # A new vertex v must connect to a subset S of existing vertices
            # such that no C4 is created.
            
            success = True
            for _ in range(target_n - n_base):
                new_v = n
                n += 1
                adj.append(0)
                
                # Candidate neighbors: all existing vertices
                candidates = list(range(new_v))
                random.shuffle(candidates)
                
                chosen = []
                # Greedily pick neighbors for new_v
                # We must ensure that adding new_v with neighbor set S doesn't create C4.
                # This is equivalent to: for any u, w in existing, 
                # the number of common neighbors of u and w within S is <= 1.
                
                available = set(range(new_v))
                # We need to pick a subset S of available such that no two elements in S have a common neighbor in the existing graph.
                
                # Greedy with better heuristic: pick vertex that blocks fewest others
                while available and len(chosen) < 6:  # Allow degree up to 6
                    # Score each candidate by how many others it blocks
                    best_s = None
                    best_score = -1
                    for s in list(available):
                        ns = adj[s]
                        # Calculate forbidden set
                        forb_mask = 0
                        temp = ns
                        while temp:
                            lsb = temp & -temp
                            u = lsb.bit_length() - 1
                            forb_mask |= adj[u]
                            temp ^= lsb
                        # Count how many available vertices are forbidden
                        blocked = 0
                        for v in available:
                            if v != s and (forb_mask & (1 << v)):
                                blocked += 1
                        score = len(available) - blocked  # Higher is better (blocks fewer)
                        if score > best_score:
                            best_score = score
                            best_s = s
                    if best_s is None:
                        break
                    chosen.append(best_s)
                    available.remove(best_s)
                    
                    # Find all vertices that share a neighbor with best_s
                    ns = adj[best_s]
                    forb_mask = 0
                    temp = ns
                    while temp:
                        lsb = temp & -temp
                        u = lsb.bit_length() - 1
                        forb_mask |= adj[u]
                        temp ^= lsb
                    
                    # Remove forbidden from available
                    new_available = set()
                    for v in available:
                        if not (forb_mask & (1 << v)):
                            new_available.add(v)
                    available = new_available
                    
                # Add edges
                for s in chosen:
                    adj[new_v] |= (1 << s)
                    adj[s] |= (1 << new_v)
                    edges.append([new_v, s])
                    
            # Check if this extension is valid
            if not _has_c4_bitmask(n, adj):
                if _max_independent_set_size_bitmask(n, adj, 11) < 11:
                    best_result = {'n': n, 'edges': edges.copy()}
                    best_n = n
                    found = True
                    break
                    
        if found:
            # If we found a valid graph for target_n, we can stop or try higher?
            # We are iterating down from 39. If we find 39, we stop.
            # If we find 38, we might try 39? No, we started at 39.
            # So if we found one for the current target_n, we return it?
            # Actually, we should keep the best one found so far.
            # Since we go 39, 38, ..., the first one found is the largest.
            break
            
    # Final polish: try to add more vertices to best_result if time allows
    if time.time() < deadline - 10:
        start_n = best_n
        start_adj = [0] * start_n
        start_edges = best_result['edges'].copy()
        for u, v in start_edges:
            start_adj[u] |= (1 << v)
            start_adj[v] |= (1 << u)
        
        for extra_n in range(start_n + 1, min(44, start_n + 6)):
            if time.time() > deadline - 5:
                break
            found_extra = False
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                random.seed(attempt * 777 + extra_n)
                adj = start_adj.copy()
                edges = start_edges.copy()
                n = start_n
                
                new_v = n
                n += 1
                adj.append(0)
                
                available = set(range(new_v))
                chosen = []
                while available and len(chosen) < 6:
                    best_s = None
                    best_score = -1
                    for s in list(available):
                        ns = adj[s]
                        forb_mask = 0
                        temp = ns
                        while temp:
                            lsb = temp & -temp
                            u = lsb.bit_length() - 1
                            forb_mask |= adj[u]
                            temp ^= lsb
                        blocked = 0
                        for v in available:
                            if v != s and (forb_mask & (1 << v)):
                                blocked += 1
                        score = len(available) - blocked
                        if score > best_score:
                            best_score = score
                            best_s = s
                    if best_s is None:
                        break
                    chosen.append(best_s)
                    available.remove(best_s)
                    
                    ns = adj[best_s]
                    forb_mask = 0
                    temp = ns
                    while temp:
                        lsb = temp & -temp
                        u = lsb.bit_length() - 1
                        forb_mask |= adj[u]
                        temp ^= lsb
                    
                    new_available = set()
                    for v in available:
                        if not (forb_mask & (1 << v)):
                            new_available.add(v)
                    available = new_available
                    
                for s in chosen:
                    adj[new_v] |= (1 << s)
                    adj[s] |= (1 << new_v)
                    edges.append([new_v, s])
                
                if not _has_c4_bitmask(n, adj):
                    if _max_independent_set_size_bitmask(n, adj, 11) < 11:
                        best_result = {'n': n, 'edges': edges.copy()}
                        best_n = n
                        found_extra = True
                        break
            if found_extra:
                start_n = best_n
                start_adj = [0] * start_n
                start_edges = best_result['edges'].copy()
                for u, v in start_edges:
                    start_adj[u] |= (1 << v)
                    start_adj[v] |= (1 << u)
            else:
                break
                
    return best_result

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    
    # Try to construct a large graph
    result = _construct_large_graph()
    
    # Verify the result
    if _verify_graph(result['n'], result['edges']):
        return result
    
    # Fallback: polarity graph of PG(2,5) (31 vertices, C4-free)
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    return {'n': n_base, 'edges': edges_base}

# EVOLVE-BLOCK-END
