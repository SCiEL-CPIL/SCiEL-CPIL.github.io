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

def _find_long_cycle(adj, n, rnd, max_attempts=15):
    """
    Find a long simple cycle using randomized greedy path extension.
    Returns list of vertices (no closing duplicate) or None.
    """
    best_cycle = None
    best_len = 0
    
    for attempt in range(max_attempts):
        # Pick a random start vertex with degree >= 2
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            return best_cycle
        start = rnd.choice(candidates)
        
        path = [start]
        visited = {start}
        current = start
        
        # Extend path greedily
        while True:
            # Get unvisited neighbors
            unvisited_neighbors = [v for v in adj[current] if v not in visited]
            
            if not unvisited_neighbors:
                # No more extensions possible
                break
            
            # Prefer neighbors that are not start (unless we can close a cycle)
            # But first check if we can close a cycle (if current is adjacent to start and len >= 3)
            can_close = (len(path) >= 3 and start in adj[current])
            
            if can_close:
                # We can close the cycle now. 
                # Decision: close it or try to extend further?
                # For "longest cycle", we might want to extend more if possible.
                # But if we extend, we might not be able to close.
                # Strategy: with some probability, close. Otherwise, extend.
                # Actually, to maximize length, we should extend if possible.
                # But if we extend, we lose the ability to close via this specific path structure easily?
                # No, we can still close later.
                # However, a simple greedy "extend until stuck" often finds long paths.
                # If we are adjacent to start, closing gives a cycle of len(path).
                # If we extend, we might find a longer cycle.
                # Let's try to extend first, but keep the option to close.
                # Actually, if we extend, we move away from start. 
                # Let's just extend unless we are forced or choose to close.
                # For simplicity and speed, let's just extend greedily.
                # If we get stuck and can close, we close.
                pass
            
            # Choose next vertex
            # Heuristic: prefer vertices with lower degree (to avoid dead ends? or higher?)
            # Usually, for finding long paths, "least likely to fail" is good.
            # Let's pick randomly among unvisited neighbors, or the one with min degree.
            # Min degree is a common heuristic for Hamiltonian path/cycle heuristics.
            unvisited_neighbors.sort(key=lambda x: len(adj[x]))
            next_v = unvisited_neighbors[0]
            
            path.append(next_v)
            visited.add(next_v)
            current = next_v
            
            # If path is very long, we can stop early?
            if len(path) > n * 0.8:
                break
        
        # Check if we can close the cycle
        if len(path) >= 3 and adj[path[-1]] and path[0] in adj[path[-1]]:
            cycle = path
            if len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
    
    return best_cycle

def _decompose(edges, n, deadline, rnd):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Find a vertex with highest degree
        start = -1
        max_deg = 0
        for v in range(n):
            if len(adj[v]) > max_deg:
                max_deg = len(adj[v])
                start = v
        if start == -1:
            break
            
        # Try to find a long cycle
        cyc = _find_long_cycle(adj, n, rnd, max_attempts=5)
        
        if cyc and len(cyc) >= 3:
            # Remove cycle edges
            m = len(cyc)
            # The cycle is [v0, v1, ..., v_{m-1}]
            # Edges are (v0, v1), ..., (v_{m-2}, v_{m-1}), (v_{m-1}, v0)
            for i in range(m):
                a = cyc[i]
                b = cyc[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(cyc)
        else:
            # No cycle found, remove an edge
            if not adj[start]:
                continue
            # Pick a neighbor, preferably one with low degree
            nbrs = list(adj[start])
            nbrs.sort(key=lambda x: len(adj[x]))
            b = nbrs[0]
            adj[start].discard(b)
            adj[b].discard(start)
            remaining -= 1
            pieces.append([start, b])

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
    base_seed = 1000
    attempt = 0
    # Allocate more time per attempt to allow longer cycles to be found
    # but still do multiple restarts
    while time.time() < deadline:
        rnd = random.Random(base_seed + attempt)
        pieces = _decompose(edges, n, deadline, rnd)
        cnt = len(pieces)
        if cnt < best_count:
            best = pieces
            best_count = cnt
        attempt += 1
        # Lower bound on pieces is roughly m / max_cycle_len.
        # For random graphs, max cycle can be large.
        # If we get close to theoretical lower bound, stop.
        # m ~ n*p. Max piece size ~ n. So min pieces ~ p.
        # Let's use a heuristic stop condition.
        if best_count <= (n * 0.3) // 3: # Very optimistic
            break
        if attempt > 500:
            break
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
