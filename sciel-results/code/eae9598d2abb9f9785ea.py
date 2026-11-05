import time
import random
import sys

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _has_cycle(adj, n):
    """Check if graph has any cycle using DFS."""
    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            if _dfs_cycle(i, -1, visited, adj):
                return True
    return False

def _dfs_cycle(node, parent, visited, adj):
    visited[node] = True
    for neighbor in adj[node]:
        if not visited[neighbor]:
            if _dfs_cycle(neighbor, node, visited, adj):
                return True
        elif neighbor != parent:
            return True
    return False

def _find_long_cycle(adj, n, rng, attempts):
    best_cycle = None
    best_len = 0
    
    for _ in range(attempts):
        starts = [v for v in range(n) if adj[v]]
        if not starts:
            return None
        start = rng.choice(starts)
        
        path = [start]
        visited = {start}
        current = start
        
        while True:
            neighbors = list(adj[current])
            unvisited = [v for v in neighbors if v not in visited]
            
            if not unvisited:
                if start in adj[current] and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                break
            else:
                # Heuristic: Lookahead
                scored = []
                for v in unvisited:
                    # Count unvisited neighbors of v
                    opts = 0
                    for w in adj[v]:
                        if w not in visited:
                            opts += 1
                    # Add randomness to avoid deterministic loops
                    score = opts + rng.random() * 1.5
                    scored.append((score, v))
                
                scored.sort(reverse=True)
                next_v = scored[0][1]
                
                path.append(next_v)
                visited.add(next_v)
                current = next_v

    return best_cycle

def _decompose(edges, n, deadline, rng):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    pieces = []
    remaining_edges = len(edges)
    
    while remaining_edges > 0:
        if time.time() > deadline:
            break
            
        # Determine attempts based on time left
        time_left = deadline - time.time()
        if time_left > 10.0:
            attempts = 300
        elif time_left > 5.0:
            attempts = 100
        else:
            attempts = 20
            
        cycle = _find_long_cycle(adj, n, rng, attempts)
        
        if cycle:
            m = len(cycle)
            for i in range(m):
                u = cycle[i]
                v = cycle[(i + 1) % m]
                adj[u].discard(v)
                adj[v].discard(u)
            remaining_edges -= m
            pieces.append(cycle)
        else:
            # Verify if there is actually a cycle
            if _has_cycle(adj, n):
                # We missed it. Try a few more times with different rng or just add edges?
                # Adding edges is safer for time, but suboptimal.
                # Let's try 10 more times.
                for _ in range(10):
                    cycle = _find_long_cycle(adj, n, rng, 10)
                    if cycle:
                        m = len(cycle)
                        for i in range(m):
                            u = cycle[i]
                            v = cycle[(i + 1) % m]
                            adj[u].discard(v)
                            adj[v].discard(u)
                        remaining_edges -= m
                        pieces.append(cycle)
                        break
                else:
                    # Still no cycle found, add edges
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                pieces.append([u, v])
                                adj[u].discard(v)
                                adj[v].discard(u)
                    remaining_edges = 0
                    break
            else:
                # No cycle, add all edges
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            pieces.append([u, v])
                            adj[u].discard(v)
                            adj[v].discard(u)
                remaining_edges = 0
                break

    return pieces

def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)
        
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
            
        rng = random.Random(12345 + i)
        pieces = _decompose(edges, n, graph_deadline, rng)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
            
    return results
