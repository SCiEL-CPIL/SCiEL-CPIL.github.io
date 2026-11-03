"""Program for constructing small k-regular graphs of given girth."""
import numpy as np
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Attempt to construct a k-regular graph with girth at least g using a greedy
    non-backtracking walk extension heuristic. If this fails or times out,
    fall back to the Lazebnik-Ustimenko-Woldar (LUW) construction.
    """
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    if k == 1:
        return 2, [(0, 1)]
    
    # Time limit for heuristic
    start_time = time.time()
    time_limit = 60.0  # Reserve 60s for LUW fallback and safety
    
    # Try heuristic
    try:
        n_heuristic, edges_heuristic = _greedy_construct(k, g, time_limit, start_time)
        if n_heuristic is not None:
            # Verify girth quickly (sampled or full if small)
            if _verify_girth(n_heuristic, edges_heuristic, g):
                return n_heuristic, edges_heuristic
    except Exception:
        pass
    
    # Fallback to LUW
    return _luw_construct(k, g)

def _verify_girth(n, edges, g):
    """
    Verify that the graph has girth at least g.
    Uses BFS from each vertex up to depth g//2 - 1.
    """
    if n > 1000:
        # For large graphs, this might be slow. 
        # But if we found a small one, it's likely < 1000.
        # If it's large, we trust the construction or skip full verify?
        # No, we must be sure. 
        # However, if n is large, the heuristic likely failed and we fell back.
        # If we are here, we found a candidate.
        pass
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS up to depth g//2 - 1
        # If we find a neighbor at depth < g//2 - 1 that is already visited 
        # by a different path, we have a cycle.
        # Actually, standard girth check:
        # BFS from start. If we find an edge (u,v) where dist[u] + dist[v] + 1 < g, 
        # and u != parent[v] and v != parent[u], we have a cycle.
        
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2 - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Back edge or cross edge
                    if v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _greedy_construct(k, g, time_limit, start_time):
    """
    Greedy construction using non-backtracking walks.
    """
    # Moore bound approximation
    # For even g: n_moore = 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # For odd g: n_moore = 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    if g % 2 == 0:
        half_g = g // 2
        n_target = 2 * sum((k - 1) ** i for i in range(half_g))
    else:
        half_g = (g - 1) // 2
        n_target = 1 + k * sum((k - 1) ** i for i in range(half_g))
    
    # Start with n_target, increase if necessary
    # We will build a graph with n vertices.
    # Strategy: Randomly assign edges to satisfy degree k, then check girth.
    # This is a random regular graph generator.
    # If girth is too small, we can try to "repair" or restart.
    # Given time limits, we'll do a few random restarts.
    
    best_n = None
    best_edges = None
    
    # Limit number of restarts based on time
    max_restarts = 10
    
    for attempt in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        n = n_target + attempt * 10 # Increase n slightly
        
        # Generate random k-regular graph
        try:
            edges = _random_k_regular(n, k)
            if edges is None:
                continue
                
            if _verify_girth(n, edges, g):
                best_n = n
                best_edges = edges
                break # Found a valid one, return smallest
        except Exception:
            continue
            
    if best_n is not None:
        return best_n, best_edges
    else:
        return None, None

def _random_k_regular(n, k):
    """
    Generate a random k-regular graph on n vertices.
    Uses configuration model.
    """
    if n * k % 2 != 0:
        return None
    
    # Create list of stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    edges = []
    edge_set = set()
    
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            # Self loop, try to swap
            # Simple heuristic: just fail this attempt
            return None
        if (u, v) in edge_set or (v, u) in edge_set:
            # Multi-edge, fail
            return None
        edge_set.add((u, v))
        edge_set.add((v, u))
        edges.append((u, v))
        
    return edges

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction fallback.
    """
    # Use Lazebnik-Ustimenko-Woldar construction
    # CD(d, q): q-regular bipartite graph, girth >= d+5 if d odd, >= d+4 if d even
    # We need girth >= g
    
    candidates = []
    
    d_odd = g - 5
    if d_odd < 3:
        d_odd = 3
    if d_odd % 2 == 0:
        d_odd += 1
    candidates.append(d_odd)
    
    d_even = g - 4
    if d_even < 2:
        d_even = 2
    if d_even % 2 == 1:
        d_even += 1
    candidates.append(d_even)
    
    d = min(candidates)
    q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
