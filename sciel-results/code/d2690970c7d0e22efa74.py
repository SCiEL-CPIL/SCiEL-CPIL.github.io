"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 115  # safety margin before 120s kill
    
    # Moore bound for (k, g): for even girth g=2g', n >= 2 * sum_{i=0}^{g'-1} (k-1)^i
    if g % 2 == 0:
        gprime = g // 2
        moore = 2 * sum((k-1)**i for i in range(gprime))
    else:
        gprime = (g - 1) // 2
        moore = 1 + k * sum((k-1)**i for i in range(gprime))
    
    # n must be even for 3-regular, and >= moore
    n_min = moore
    if n_min % 2 != 0:
        n_min += 1
    if n_min < 4:
        n_min = 4
    
    best_n = None
    best_edges = None
    
    # Try increasing n from moore bound upward
    # For random 3-reg graphs, girth grows like 2*log_2(n) roughly
    # For girth 16, we need n such that 2*log_2(n) >= 16, so n >= 2^8 = 256
    # But Moore bound is tighter. Let's try a range.
    
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Try different n values: start near moore bound, increase
        # Random 3-reg graphs: expected shortest cycle ~ 2*log_{k-1}(n)
        # For k=3: ~2*log_2(n). For girth 16: need 2*log_2(n) >= 16 => n >= 256
        # But we need to actually verify. Let's try n values.
        
        # Start from moore bound, try a few candidates
        if attempt < 50:
            n = n_min
        elif attempt < 100:
            n = n_min + 2 * ((attempt - 50) // 5)
        elif attempt < 150:
            n = n_min + 2 * (10 + (attempt - 100) // 5)
        else:
            n = n_min + 2 * (20 + (attempt - 150) // 5)
        
        if n < 4 or n % 2 != 0:
            n += 1
        if n < 4:
            n = 4
        
        # Try to construct a random 3-regular graph on n vertices
        edges = _random_3_regular(n)
        if edges is None:
            continue
        
        # Verify girth >= g using BFS from each vertex (bounded depth)
        if _check_girth(n, edges, g):
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
                # If we hit the Moore bound, that's optimal
                if n == n_min:
                    break
            # If we found a valid graph, try smaller n next
            # Actually, let's just keep trying to find smaller ones
        else:
            # Girth too small, try larger n
            pass
    
    if best_n is None:
        # Fallback: use the original LUDW construction
        return _ludw_fallback(k, g)
    
    return best_n, best_edges

def _random_3_regular(n):
    """Generate a random 3-regular simple graph on n vertices (n must be even)."""
    if n < 4 or n % 2 != 0:
        return None
    
    for trial in range(50):
        # Use configuration model with rejection
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        
        if not valid:
            continue
        
        # Check connectivity (optional, but good)
        # For girth purposes, we just need the graph
        edge_list = list(edges)
        
        # Verify it's actually 3-regular
        deg = [0] * n
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            continue
        
        return edge_list
    
    return None

def _check_girth(n, edges, g):
    """Check if the graph has girth >= g (no cycle shorter than g)."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from each vertex, bounded depth g-1
    # If we find a vertex at distance < g/2 that connects back, girth < g
    # More precisely: do BFS from each vertex up to depth floor((g-1)/2)
    # If any vertex is reachable from itself via a non-backtracking walk of length < g, girth < g
    
    # Simpler: for each vertex, BFS to depth (g-1)//2, check if we revisit the start
    # Actually, standard approach: for each vertex s, do BFS. If s is reached at depth d < g, 
    # then there's a cycle of length d (if d is even) or we need to be more careful.
    
    # Correct approach: The girth is the length of the shortest cycle.
    # A cycle of length L means there's a closed non-backtracking walk of length L.
    # For BFS from s: if we reach s at depth d, and d >= 2, then there's a cycle of length <= d.
    # But BFS finds shortest path, so if s is reached at depth d, there's a cycle of length d.
    # Wait, no. BFS from s: the first time we reach s (other than the start), that's the girth?
    # Not exactly. Let me think again.
    
    # Standard method: For each vertex s, do BFS. Track parent. If we find an edge (u,v) where 
    # u and v are both at depth d and are not parent-child, then we have a cycle of length 2*d.
    # Or if v is an ancestor of u, we have a cycle.
    
    # Simpler: just do BFS from each vertex, and if we can reach the start vertex at depth < g, 
    # and the path doesn't immediately backtrack, then girth < g.
    
    # Let me use a cleaner approach: for each vertex s, BFS up to depth g-1. 
    # If we encounter s again (via a different path), the cycle length is the depth.
    
    half_g = (g - 1) // 2  # We only need to search this deep
    
    for s in range(n):
        # BFS from s, up to depth half_g
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if dist[u] >= half_g:
                continue
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < half_g:
                        queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle: u-v is a cross edge
                    # Cycle length is dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _ludw_fallback(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
