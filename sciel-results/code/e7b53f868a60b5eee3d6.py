import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    # 120s limit, keep 10s for safety/fallback
    deadline = start_time + 110
    
    # k is expected to be 3, g to be 16
    # But handle generic k, g if possible?
    # Task says "evaluator passes this task's degree k and girth g".
    # It's specifically the 3,16 cage.
    
    if k != 3:
        # Fallback to LUW for other k if needed, but task is specific
        return _luw_construct(k, g)

    # Target N values
    # Moore bound ~ 510.
    # Try a range.
    # If we find a valid graph, we return it.
    # We want the smallest N.
    
    # We will try N in increasing order.
    # Time budget per N: 2 seconds.
    # Max N to try: 2000.
    
    best_result = None
    
    # Generate candidate Ns
    # Start at 512, step by 16? Or just a list.
    # Let's use a list of even numbers from 512 to 1500.
    candidates = list(range(512, 1501, 16))
    
    for N in candidates:
        if time.time() > deadline - 15: # 15s reserved for fallback
            break
            
        # Try to find graph of size N with 2s budget
        res = _search_cubic_graph(N, g, time.time() + 2)
        if res:
            best_result = res
            break
            
    if best_result:
        return best_result
        
    # If no graph found in the search range, try larger N with more time?
    # Or just fallback to LUW.
    # Actually, if we didn't find one up to 1500, it's unlikely we will find one
    # quickly with random search. LUW is safe.
    
    return _luw_construct(k, g)

def _search_cubic_graph(N, g, deadline):
    """
    Tries to construct a 3-regular graph of size N with girth >= g.
    Returns (N, edges) or None.
    """
    if N % 2 != 0:
        return None
        
    # 1. Generate random 3-regular graph
    adj = _generate_random_cubic(N)
    if adj is None:
        return None
        
    edges = _adj_to_edges(adj)
    
    # 2. Local search
    while time.time() < deadline:
        girth, cycle = _calculate_girth_and_cycle(adj, g)
        if girth >= g:
            return N, edges
            
        if cycle is None:
            break # Should not happen if girth < g
            
        # Break the cycle by swapping an edge in the cycle
        # Pick random edge in cycle
        e_idx = random.randint(0, len(cycle) - 1)
        u, v = cycle[e_idx]
        
        # Find a random edge to swap with
        # Must be non-adjacent to u and v to avoid loops/multiedges
        # And not in the cycle (ideally)
        swapped = False
        for _ in range(50): # Try 50 times to find a valid swap
            # Pick random edge from graph
            # We need a list of edges for fast random pick
            # Re-generating edge list is slow. Keep it updated?
            # For N=1000, edges=1500. List is small.
            all_edges = _adj_to_edges(adj)
            # Filter edges
            candidates = []
            for x, y in all_edges:
                if x == u or x == v or y == u or y == v:
                    continue
                if (x, y) == (u, v) or (y, x) == (u, v):
                    continue
                candidates.append((x, y))
                
            if not candidates:
                continue
                
            x, y = random.choice(candidates)
            
            # Perform swap: remove (u,v) and (x,y), add (u,x) and (v,y)
            # Check if (u,x) or (v,y) already exists
            if _has_edge(adj, u, x) or _has_edge(adj, v, y):
                continue
            if _has_edge(adj, u, y) or _has_edge(adj, v, x):
                continue
                
            # Remove
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            # Add
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            edges = _adj_to_edges(adj)
            swapped = True
            break
            
        if not swapped:
            # Try a different edge in cycle or random restart?
            # Just pick another edge in cycle
            continue
            
    return None

def _generate_random_cubic(N):
    """Generates a random 3-regular graph on N vertices."""
    if N < 4 or N % 2 != 0:
        return None
        
    stubs = []
    for i in range(N):
        stubs.extend([i, i, i])
        
    random.shuffle(stubs)
    
    adj = [[] for _ in range(N)]
    edges = []
    
    # Pair up stubs
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        
        if u == v:
            # Loop, retry logic needed or restart
            # Simple restart for small N
            return _generate_random_cubic(N)
            
        if v in adj[u]:
            # Multi-edge, retry
            return _generate_random_cubic(N)
            
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        
    return adj

def _adj_to_edges(adj):
    edges = []
    n = len(adj)
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return edges

def _has_edge(adj, u, v):
    return v in adj[u]

def _calculate_girth_and_cycle(adj, target_girth):
    """
    Calculates girth and returns the shortest cycle found.
    Returns (girth, cycle_edges)
    """
    n = len(adj)
    best_girth = float('inf')
    best_cycle = None
    
    # For each edge, find shortest path between endpoints in G - edge
    for u in range(n):
        for v in adj[u]:
            if u > v: continue # Each edge once
            
            # BFS from u to v, avoiding edge (u,v)
            # dist, parent
            dist = [-1] * n
            parent = [-1] * n
            queue = [u]
            dist[u] = 0
            
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor == v and curr == u:
                        continue # Skip the edge we removed
                    if neighbor == u and curr == v:
                        continue
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                        
            if dist[v] != -1:
                cycle_len = dist[v] + 1
                if cycle_len < best_girth:
                    best_girth = cycle_len
                    # Reconstruct cycle
                    cycle = []
                    curr = v
                    while curr != u:
                        p = parent[curr]
                        cycle.append((p, curr))
                        curr = p
                    cycle.append((u, v))
                    best_cycle = cycle
                    
                if best_girth >= target_girth:
                    return best_girth, best_cycle # Early exit
            
            if best_girth >= target_girth:
                return best_girth, best_cycle
                
    return best_girth, best_cycle

def _luw_construct(k, g):
    # Original seed code logic
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
