import time
import random
import sys
import math
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses a combination of a known algebraic construction (Lazebnik-Ustimenko-Woldar) as a fallback
    and a heuristic search (Randomized Edge Swapping) to find smaller graphs.
    """
    # Deadline: 110 seconds to be safe within 120s limit
    deadline = time.time() + 110.0
    
    # 1. Compute the fallback LUW graph
    n_luw, edges_luw = _construct_luw(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # If the fallback is already small, we might not need to search much, 
    # but let's try to improve.
    
    # Moore bound for n(k, g)
    if k == 2:
        moore = g
    else:
        # Moore bound for odd girth g: 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        # For g=16, it's even, so we look for girth >= 16.
        # The Moore bound for even girth 2r is 2 * sum_{i=0}^{r-1} (k-1)^i
        # For k=3, g=16 (r=8): 2 * (1 + 2 + 4 + ... + 128) = 2 * 255 = 510.
        # Wait, the prompt says lower bound is 512.
        # Let's just use 512 for k=3, g=16.
        moore = 512 if (k == 3 and g == 16) else _moore_bound(k, g)
        
    # Ensure n is even for k=3
    if moore % 2 != 0:
        moore += 1
        
    # Search range: from Moore bound to best_n - 2
    # Step size: 2 (since n must be even for 3-regular)
    current_n = moore
    
    # Time check
    if time.time() > deadline - 5:
        return best_n, best_edges
        
    # Heuristic Search
    # We try to find a valid graph for n < best_n
    # We iterate n upwards. If we find one, we update best and continue to see if we can go smaller?
    # No, we want the MINIMUM n. So we should search from Moore bound UP.
    # If we find a valid graph at n=520, that's better than 936.
    # Do we search for n=512, 514, ...?
    # Yes.
    
    max_search_n = best_n - 2
    if max_search_n < moore:
        return best_n, best_edges
        
    # To be efficient, we might skip some n or limit attempts per n.
    # Let's try a few random restarts for each n.
    
    while current_n <= max_search_n and time.time() < deadline - 5:
        # Try to find a valid graph of size current_n
        # Limit time per n to avoid getting stuck
        time_limit_for_n = min(10.0, deadline - time.time() - 5)
        
        found = _try_find_graph(k, g, current_n, time_limit_for_n)
        if found:
            n_found, edges_found = found
            if n_found < best_n:
                best_n = n_found
                best_edges = edges_found
                # If we found one at the Moore bound, we can stop? 
                # No, maybe there's a smaller one? No, Moore bound is the theoretical min.
                # So if we find one at moore, it's optimal.
                if current_n == moore:
                    break
            # If we found one, we can try to find an even smaller one? 
            # No, we are iterating UP from Moore. 
            # So the first one we find is the smallest in this sequence.
            # But we skipped some n? No, we step by 2.
            # So if we find one at 520, we know 512 and 514 failed (or we didn't try hard enough).
            # We should probably stop searching for larger n if we found a good one, 
            # UNLESS we want to ensure we didn't miss a smaller one due to insufficient search time.
            # Given the time constraint, finding ANY valid graph < LUW is a big win.
            # Let's stop searching for larger n once we find one, as going larger increases n (worse score).
            # We only want to go SMALLER. Since we start at Moore, the first success is the best.
            break
            
        # If not found, try next n
        current_n += 2

    return best_n, best_edges

def _construct_luw(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    # d is the parameter such that girth >= d+5 (for odd d) or similar.
    # For g=16, we need girth >= 16.
    # The LUW graph CD(d, q) has girth 2*floor((d+1)/2) + 4 ? 
    # Let's use the logic from the seed program which worked.
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
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _moore_bound(k, g):
    if k == 2:
        return g
    if g % 2 == 0:
        r = g // 2
        # 2 * sum_{i=0}^{r-1} (k-1)^i
        s = 0
        term = 1
        for i in range(r):
            s += term
            term *= (k - 1)
        return 2 * s
    else:
        r = (g - 1) // 2
        # 1 + k * sum_{i=0}^{r-1} (k-1)^i
        s = 0
        term = 1
        for i in range(r):
            s += term
            term *= (k - 1)
        return 1 + k * s

def _try_find_graph(k, g, n, time_limit):
    """
    Attempts to find a k-regular graph with girth >= g on n vertices.
    Returns (n, edges) if found, else None.
    """
    if n % 2 != 0:
        return None
        
    start_time = time.time()
    max_attempts = 50  # Number of random restarts
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        # Generate random k-regular graph
        edges = _generate_random_k_regular(n, k)
        if edges is None:
            continue
            
        # Check if girth is sufficient
        if _has_girth_at_least(n, edges, g):
            return n, edges
            
        # Try to improve via edge swaps
        edges = _improve_girth(n, edges, k, g, time_limit - (time.time() - start_time))
        
        if edges and _has_girth_at_least(n, edges, g):
            return n, edges
            
    return None

def _generate_random_k_regular(n, k):
    """
    Generates a random k-regular graph using configuration model.
    Returns list of edges or None if failed to generate simple graph.
    """
    if n < k or (n * k) % 2 != 0:
        return None
        
    # Create stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            continue # Loop, skip or retry? For simplicity, if we get too many, fail.
            # Actually, we should reject and regenerate if loops/multiedges are frequent.
            # For k=3, probability is low.
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            # Multiedge, fail this attempt
            return None
        edges.add((u, v))
        
    if len(edges) != n * k // 2:
        return None
        
    return list(edges)

def _has_girth_at_least(n, edges, g):
    """
    Checks if the graph has girth at least g.
    Uses BFS from each vertex to find shortest cycles.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for cycles of length < g
    # For each node, do BFS up to depth g-2
    # If we find an edge (u, v) where dist(u, v) < g-1, we have a cycle < g
    # Actually, if dist(u, v) = d, and (u,v) is an edge, cycle length is d+1.
    # We need d+1 >= g => d >= g-1.
    # So if we find any edge (u,v) such that shortest path u->v is < g-1, it's invalid.
    
    # Optimization: We can just check if there is ANY cycle of length < g.
    # BFS from each node u:
    #   dist[u] = 0
    #   For neighbors v:
    #      if dist[v] is not set, set dist[v] = 1, parent[v] = u
    #      if dist[v] is set and v != parent[u] and dist[u] + dist[v] + 1 < g:
    #          # Found a cycle of length dist[u] + dist[v] + 1
    #          return False
    #   ...
    # This is O(n * (n + m)) which is O(n^2) for sparse graphs.
    # For n=512, 512^2 = 262k. Very fast.
    
    for u in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[u] = 0
        queue = deque([u])
        
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif neighbor != parent[node]:
                    # Found a back edge
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < g:
                        return False
                        
    return True

def _improve_girth(n, edges, k, g, time_limit):
    """
    Attempts to increase girth by swapping edges.
    """
    if not edges:
        return None
        
    start_time = time.time()
    adj = _build_adj(n, edges)
    
    # Find a short cycle to break
    # We can randomly pick edges and check if they are in a short cycle.
    # Or find the shortest cycle.
    
    # For simplicity, we just do random swaps and check if girth improves.
    # This is slow.
    
    # Better: Find a specific short cycle.
    # Let's find one short cycle.
    short_cycle = _find_short_cycle(n, adj, g)
    
    if not short_cycle:
        return edges # Already good
        
    # Try to break it
    # Pick an edge in the cycle.
    # Try to swap it with a random other edge.
    
    # This is a placeholder for a more sophisticated heuristic.
    # Given the time, we might just return the original edges if we can't quickly improve.
    # Or we can do a few random swaps.
    
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random edge
        u, v = random.choice(edges)
        # Pick a random other edge
        x, y = random.choice(edges)
        
        # Skip if same edge or adjacent
        if {u, v} == {x, y}:
            continue
        if v == x or v == y or u == x or u == y:
            continue
            
        # Swap (u,v) and (x,y) to (u,x) and (v,y)
        new_edges = set(edges)
        new_edges.remove((u, v) if u < v else (v, u))
        new_edges.remove((x, y) if x < y else (y, x))
        
        # Check for loops/multiedges in new edges
        if u == x or v == y or u == y or v == x:
            continue
            
        e1 = (u, x) if u < x else (x, u)
        e2 = (v, y) if v < y else (y, v)
        
        if e1 in new_edges or e2 in new_edges:
            continue
            
        new_edges.add(e1)
        new_edges.add(e2)
        
        # Check if new graph is valid (simple, regular) and has better girth
        # We don't want to decrease girth.
        # So we check if girth(new) >= girth(old) or simply if girth(new) >= g.
        # If girth(new) >= g, we accept.
        
        new_edge_list = list(new_edges)
        
        # Quick check: is it still k-regular?
        # Swapping two edges in a k-regular graph preserves k-regularity if no loops/multiedges.
        
        # Check girth
        if _has_girth_at_least(n, new_edge_list, g):
            return new_edge_list
        # Else, reject.
            
    return edges

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_short_cycle(n, adj, g):
    """
    Finds a cycle of length < g. Returns the list of nodes in the cycle.
    """
    # BFS from each node
    for u in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[u] = 0
        queue = deque([u])
        
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif neighbor != parent[node]:
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = []
                        cur = node
                        while cur != neighbor:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.append(neighbor)
                        # Reverse to get order from neighbor to node?
                        # The path is u -> ... -> neighbor and u -> ... -> node.
                        # The cycle is node -> ... -> u -> ... -> neighbor -> node?
                        # Actually, the back edge is (node, neighbor).
                        # Path 1: u to node. Path 2: u to neighbor.
                        # Cycle: node -> ... -> u -> ... -> neighbor -> node.
                        # Let's just return the nodes involved.
                        return cycle
    return None
# EVOLVE-BLOCK-END
