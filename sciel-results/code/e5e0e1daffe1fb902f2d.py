import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose_greedy(edges, n, rnd):
    """
    Greedy decomposition: repeatedly find a long cycle, remove it, until no cycles remain.
    Remaining edges become single-edge pieces.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)

    # To speed up, we can limit the number of cycles we try to find
    # But for these small graphs, we should be able to find many.
    
    while remaining > 0:
        # Check if any cycle exists (simple check: any vertex with degree >= 2)
        # Actually, a graph with min degree >= 2 contains a cycle.
        # If min degree < 2, it might still have cycles, but let's just try to find one.
        
        # Find a cycle using randomized DFS
        cycle = None
        # Try a few random starts
        for _ in range(20):
            # Pick a random vertex with degree >= 2
            candidates = [v for v in range(n) if len(adj[v]) >= 2]
            if not candidates:
                break
            start = rnd.choice(candidates)
            
            # DFS to find a cycle
            # We want a simple cycle.
            # We can do a DFS from start, and if we hit start again, we found a cycle.
            # To avoid long paths that don't close, we can limit depth or just let it run.
            # Given n <= 30, DFS is fast.
            
            stack = [(start, [start])]
            visited_in_dfs = {start}
            # We need to track the path to reconstruct the cycle.
            # Standard DFS for cycle detection in undirected graph:
            # If we see a visited neighbor that is not the parent, we have a cycle.
            # But we need the actual cycle vertices.
            
            # Let's use a different approach: Randomized greedy path extension
            path = [start]
            visited = {start}
            current = start
            parent = {start: -1}
            
            while True:
                # Find unvisited neighbor
                unvisited = [v for v in adj[current] if v not in visited]
                if not unvisited:
                    break
                
                # Check if we can close the cycle
                if start in adj[current] and len(path) >= 3:
                    # Found a cycle
                    cycle = path[:]
                    break
                
                # Choose next vertex
                # Heuristic: choose neighbor with fewest unvisited neighbors (Warnsdorff's rule variant?)
                # Or just random.
                # Let's try to extend the path.
                # Sort by degree in remaining graph to avoid dead ends?
                # Actually, for finding *a* cycle, random is fine.
                # For finding a *long* cycle, maybe prefer low degree?
                unvisited.sort(key=lambda x: len(adj[x]))
                next_v = unvisited[0]
                
                path.append(next_v)
                visited.add(next_v)
                parent[next_v] = current
                current = next_v
            
            if cycle:
                break
        
        if cycle:
            # Remove cycle edges
            m = len(cycle)
            for i in range(m):
                a = cycle[i]
                b = cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(cycle)
        else:
            # No cycle found. Remove an edge.
            # Find any edge
            edge_found = False
            for u in range(n):
                if adj[u]:
                    v = next(iter(adj[u]))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    pieces.append([u, v])
                    remaining -= 1
                    edge_found = True
                    break
            if not edge_found:
                break

    # Leftover edges
    for u in range(n):
        for v in list(adj[u]):
            if u < v:
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])

    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    base_seed = 42
    attempt = 0
    
    # We will run multiple restarts with different random seeds
    # and keep the best result
    while time.time() < deadline - 0.5:
        rnd = random.Random(base_seed + attempt)
        pieces = _decompose_greedy(edges, n, rnd)
        cnt = len(pieces)
        if cnt < best_count:
            best = pieces
            best_count = cnt
        
        attempt += 1
        # Stop if we've tried enough or time is running out
        if attempt > 200:
            break
        
        # Heuristic: if we've found a solution close to the theoretical minimum, stop early?
        # Theoretical minimum is roughly m / max_cycle_len.
        # For random graphs, max cycle is often close to n.
        # So min pieces is roughly m/n.
        # If best_count is close to that, we can stop.
        # But for now, just let it run until deadline.
        
    return best

def construct():
    start_time = time.time()
    total_budget = 140.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # Give more time to larger graphs if needed, but keep balanced
        graph_deadline = start_time + per_graph_budget * (i + 1)
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
        
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
    return results
