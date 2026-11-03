"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

import numpy as np

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if k == 4 and g == 10:
        # The LUW construction gives 200 vertices for (4,10).
        # Try to find something smaller via random search + local search.
        # Target n values: must be even (4-regular requires n*4 even),
        # and n >= Moore bound. For (4,10): Moore bound = 2 + 4*(1+3+9+27) = 2+4*40 = 162.
        # But girth 10 means no cycles of length 2,3,4,5,6,7,8,9.
        # The tree of depth 4 from a vertex has 1+4+12+36+108 = 161 vertices,
        # so n >= 162. But the graph must be 4-regular.
        
        # Start from n=162 (even, >= Moore bound) up to 200 (LUW result)
        # Try random search for smaller n first, then fall back to LUW.
        
        target_ns = list(range(162, 201, 2))
        
        best_result = None
        best_n = 201  # start above LUW so LUW is always better
        
        for n in target_ns:
            if time.time() > deadline - 5:
                break
            
            # Try random search with local improvement
            result = _search_for_n(n, k, g, deadline)
            if result is not None:
                best_n = n
                best_result = result
                break  # found the smallest n, return it
        
        if best_result is not None:
            return best_result
        
        # Fallback: LUW construction gives n=200
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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


def _search_for_n(n, k, g, deadline):
    """Try to find a k-regular graph with girth >= g on n vertices."""
    if n * k % 2 != 0:
        return None
    if n < k + 1:
        return None
    
    max_attempts = 50
    rng = random.Random()
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 3:
            return None
        
        # Generate random k-regular graph
        adj = _gen_random_kreg(n, k, rng)
        if adj is None:
            continue
        
        # Check girth
        girth = _compute_girth(adj, n, g)
        if girth >= g:
            edges = _adj_to_edges(adj, n)
            return n, edges
        
        # If girth is close, try local search to improve
        if girth >= g - 2:
            adj, girth = _local_search(adj, n, k, g, deadline, rng)
            if girth >= g:
                edges = _adj_to_edges(adj, n)
                return n, edges
    
    return None


def _gen_random_kreg(n, k, rng):
    """Generate a random k-regular graph on n vertices using configuration model.
    Returns adjacency list or None if failed."""
    if n * k % 2 != 0:
        return None
    
    # Configuration model: create n*k stubs, pair them randomly
    for _ in range(50):
        stubs = []
        for v in range(n):
            for _ in range(k):
                stubs.append(v)
        
        rng.shuffle(stubs)
        
        # Pair consecutive stubs
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if valid:
            # Convert to lists
            adj = [list(s) for s in adj]
            return adj
    
    return None


def _compute_girth(adj, n, g_limit):
    """Compute the girth of the graph. Returns girth or a value >= g_limit if girth >= g_limit.
    Uses BFS from each vertex, looking for cycles of length < g_limit."""
    # Fast check: use BFS from each vertex, track distances
    # For girth >= g, no cycle of length < g exists.
    # A cycle of length L through start vertex s: in BFS from s, 
    # if we find an edge (u,v) where both are at distance d from s and 
    # they're not parent-child, we get a cycle of length 2*d+1.
    # More precisely: for each start s, BFS. For each edge (u,v) not in BFS tree,
    # cycle length = dist[u] + dist[v] + 1. If this < g_limit, girth < g_limit.
    
    # Optimization: we only need to check up to depth (g_limit-1)//2
    max_depth = (g_limit - 1) // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
        
        # Check all edges for short cycles
        for u in range(n):
            for v in adj[u]:
                if u > v:
                    continue
                if parent[u] == v or parent[v] == u:
                    continue  # tree edge
                if dist[u] != -1 and dist[v] != -1:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g_limit:
                        return cycle_len  # found a short cycle
    
    return g_limit  # no short cycle found


def _local_search(adj, n, k, g, deadline, rng):
    """Try to increase girth by flipping edges. 
    Pick a short cycle and try to break it by swapping edges."""
    # Find a short cycle
    for iteration in range(200):
        if time.time() > deadline - 3:
            break
        
        girth = _compute_girth(adj, n, g)
        if girth >= g:
            return adj, girth
        
        # Find a cycle of length < g
        cycle = _find_short_cycle(adj, n, g)
        if cycle is None:
            return adj, girth
        
        # Try to break the cycle by edge swaps
        # For a 4-regular graph, we can do 2-opt style swaps
        # If we have edges (a,b) and (c,d) not in the cycle, 
        # we can swap to (a,c) and (b,d) or (a,d) and (b,c)
        
        # Simpler: pick two vertices on the cycle and try to reconnect
        # Actually, let's just try random edge swaps
        for _ in range(10):
            # Pick a random edge to remove
            u = rng.randint(0, n - 1)
            v = rng.choice(adj[u])
            
            # Find another edge to swap with
            # Remove (u,v), then for each neighbor of u and neighbor of v,
            # try adding (u, v2) and (v, u2)
            
            # Save original
            adj_u_orig = list(adj[u])
            adj_v_orig = list(adj[v])
            
            # Remove edge (u,v)
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Try all combinations
            best_swap = None
            for u2 in adj[u]:
                for v2 in adj[v]:
                    if u2 == v or v2 == u:
                        continue
                    if u2 == v2:
                        continue
                    if v2 in adj[u] or u2 in adj[v]:
                        continue
                    # Try adding (u, v2) and (v, u2)
                    adj[u].append(v2)
                    adj[v].append(u2)
                    adj[u2].append(u)
                    adj[v2].append(v)
                    
                    # Check if this improves girth (quick check)
                    # Just check if the specific cycle is broken
                    new_girth = _compute_girth(adj, n, g)
                    
                    if new_girth > girth:
                        best_swap = (u, v2, v, u2)
                        break
                    else:
                        # Undo
                        adj[u].remove(v2)
                        adj[v].remove(u2)
                        adj[u2].remove(u)
                        adj[v2].remove(v)
                
                if best_swap:
                    break
            
            if best_swap:
                # Already applied the best swap
                girth = _compute_girth(adj, n, g)
                continue
            else:
                # Restore original
                adj[u] = adj_u_orig
                adj[v] = adj_v_orig
    
    final_girth = _compute_girth(adj, n, g)
    return adj, final_girth


def _find_short_cycle(adj, n, g_limit):
    """Find a cycle of length < g_limit. Returns list of vertices or None."""
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g_limit - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g_limit:
                        # Reconstruct cycle
                        cycle = [v]
                        cur = v
                        while cur != start:
                            cur = parent[cur]
                            cycle.append(cur)
                        cycle.append(start)
                        return cycle
    
    return None


def _adj_to_edges(adj, n):
    """Convert adjacency list to edge list."""
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges
# EVOLVE-BLOCK-END
