import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for spec in specs:
        n, p, seed = spec
        edges = generate_graph(n, p, seed)
        
        best_decomp = None
        best_count = float('inf')
        
        if n <= 20:
            max_restarts = 500
        elif n <= 25:
            max_restarts = 200
        else:
            max_restarts = 100

        restarts_done = 0
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            rng = random.Random(r * 1000 + seed)
            decomp = greedy_decompose(edges, n, rng)
            count = len(decomp)
            
            if count < best_count:
                best_count = count
                best_decomp = decomp
                if count == 1:
                    break
            restarts_done += 1

        if best_decomp is None:
            best_decomp = [[u, v] for u, v in edges]
        
        results.append(best_decomp)

    return results

def generate_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set()
    for i in range(n - 1):
        u, v = order[i], order[i+1]
        edges.add((min(u,v), max(u,v)))
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return edges

def get_2_core(edges, n):
    """
    Returns the set of edges in the 2-core of the graph.
    """
    if not edges:
        return set()
    
    # Build degree count and adjacency
    degree = {i: 0 for i in range(n)}
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        
    # Queue of vertices with degree < 2
    q = deque([i for i in range(n) if degree[i] < 2])
    removed = set()
    
    while q:
        u = q.popleft()
        if u in removed:
            continue
        if degree[u] >= 2:
            continue
            
        removed.add(u)
        for v in list(adj[u]):
            if v in removed:
                continue
            degree[v] -= 1
            adj[v].discard(u)
            adj[u].discard(v) # Not strictly needed if we don't revisit, but good for cleanliness
            if degree[v] < 2:
                q.append(v)
                
    # Collect edges where both endpoints are not removed
    core_edges = set()
    for u, v in edges:
        if u not in removed and v not in removed:
            core_edges.add((u, v))
            
    return core_edges

def greedy_decompose(edges, n, rng):
    remaining = set(edges)
    pieces = []
    
    while remaining:
        # Find the 2-core
        core_edges = get_2_core(remaining, n)
        
        if not core_edges:
            # No cycles left, all remaining edges are bridges/leaves
            for e in remaining:
                pieces.append([e[0], e[1]])
            break
        
        # Find a cycle in the core
        # We can use the existing heuristic, but restricted to core_edges
        # To ensure we find a cycle, we can just pick any edge in the core
        # and try to find a path between its endpoints in the rest of the core.
        # Or use the DFS heuristic on the core.
        
        cycle = find_cycle_heuristic(core_edges, n, rng)
        
        if cycle:
            pieces.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.remove(e)
        else:
            # Fallback: If heuristic fails (shouldn't happen for 2-core),
            # pick any edge from core and treat as single edge?
            # Or just pick any edge from remaining.
            # Since core has min degree >= 2, there MUST be a cycle.
            # If find_cycle_heuristic fails, it's a bug or timeout.
            # Let's force a simple cycle finding:
            # Pick an edge (u,v). Find path u->v in core - (u,v).
            # This is guaranteed to exist.
            
            # For safety, if heuristic fails, just take one edge
            e = next(iter(core_edges))
            pieces.append([e[0], e[1]])
            remaining.remove(e)
            
    return pieces

def find_cycle_heuristic(remaining, n, rng):
    if not remaining:
        return None
        
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        
    active_nodes = [i for i in range(n) if adj[i]]
    if not active_nodes:
        return None
        
    for node in active_nodes:
        adj[node] = list(adj[node])
        rng.shuffle(adj[node])
        
    best_cycle = None
    best_len = 0
    
    # Since this is called on the 2-core, we know a cycle exists.
    # We can try to find a long one.
    # Limit steps to prevent hanging on large cores.
    
    for start in active_nodes:
        steps_used = 0
        max_steps = 5000 # Reduced since core is smaller/denser
        
        def dfs_shared(u, path, visited):
            nonlocal steps_used, best_cycle, best_len
            
            if steps_used >= max_steps:
                return False
            
            steps_used += 1
            
            if len(path) > 2 and start in adj[u]:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = list(path)
            
            if len(path) + (n - len(visited)) <= best_len:
                return False
            
            for v in adj[u]:
                if v == start:
                    if len(path) > 2:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                    continue
                if v not in visited:
                    visited.add(v)
                    path.append(v)
                    if not dfs_shared(v, path, visited):
                        path.pop()
                        visited.remove(v)
                        return False
                    path.pop()
                    visited.remove(v)
                    
            return True

        dfs_shared(start, [start], {start})
        
        if best_len == n:
            break
        if steps_used >= max_steps:
            break

    return best_cycle
