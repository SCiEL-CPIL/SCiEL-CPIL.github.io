"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _path_cover_greedy(edges, n, max_time=100.0):
    """
    Greedy edge-disjoint path cover.
    Returns list of paths (each path is a list of vertices).
    """
    start_time = time.time()
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute degrees
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    # Odd-degree vertices must be endpoints
    odd_vertices = [i for i in range(n) if deg[i] % 2 == 1]
    
    # We want to cover all edges with minimum number of paths.
    # A path has two endpoints. If we have k paths, we have 2k endpoints.
    # Vertices of odd degree must be endpoints.
    # So minimum number of paths >= ceil(odd_count / 2).
    
    # Strategy: Start from odd-degree vertices, extend paths greedily.
    # Also, we can start from even-degree vertices if needed.
    
    # First, let's try to form paths starting from odd vertices.
    # We'll maintain a set of available edges.
    remaining_edges = set(tuple(sorted(e)) for e in edges)
    
    paths = []
    
    # Sort odd vertices by degree (higher degree first, to make longer paths)
    odd_vertices.sort(key=lambda x: -deg[x])
    
    # We need to pair up odd vertices into paths.
    # But we don't know which odd vertices should be paired.
    # Let's just start paths from each odd vertex and see.
    
    # Actually, a better approach: 
    # 1. Identify connected components (graph is connected, so one component).
    # 2. Find a set of paths that cover all edges.
    
    # Let's use a different approach:
    # - Maintain remaining edges.
    # - While edges remain, find a path.
    # - To minimize number of paths, we want each path to be as long as possible.
    # - Start from a vertex with remaining degree > 0.
    # - Prefer starting from odd-degree vertices (in the remaining graph).
    
    # Let's track remaining degree.
    rem_deg = [0] * n
    for u, v in edges:
        rem_deg[u] += 1
        rem_deg[v] += 1
    
    # Build adjacency from remaining edges
    rem_adj = defaultdict(set)
    for u, v in edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    while remaining_edges:
        if time.time() - start_time > max_time:
            break
        
        # Find a vertex with rem_deg > 0
        # Prefer odd rem_deg vertices
        start_candidates = [i for i in range(n) if rem_deg[i] > 0 and rem_deg[i] % 2 == 1]
        if not start_candidates:
            start_candidates = [i for i in range(n) if rem_deg[i] > 0]
        
        if not start_candidates:
            break
        
        # Pick the one with highest degree
        start = max(start_candidates, key=lambda x: rem_deg[x])
        
        # Extend path from start
        path = [start]
        visited_edges = set()
        current = start
        
        while True:
            # Find a neighbor with remaining edge
            neighbors = [v for v in rem_adj[current] if (min(current, v), max(current, v)) in remaining_edges and (min(current, v), max(current, v)) not in visited_edges]
            if not neighbors:
                break
            
            # Pick the neighbor with highest rem_deg (to continue longer)
            # But avoid going back to previous vertex if possible
            prev = path[-2] if len(path) >= 2 else None
            candidates = [v for v in neighbors if v != prev]
            if candidates:
                nxt = max(candidates, key=lambda x: rem_deg[x])
            else:
                nxt = neighbors[0]
            
            # Traverse edge
            edge = (min(current, nxt), max(current, nxt))
            visited_edges.add(edge)
            remaining_edges.discard(edge)
            rem_deg[current] -= 1
            rem_deg[nxt] -= 1
            rem_adj[current].discard(nxt)
            rem_adj[nxt].discard(current)
            
            path.append(nxt)
            current = nxt
        
        if len(path) >= 2:
            paths.append(path)
    
    return paths

def construct():
    start_time = time.time()
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        paths = _path_cover_greedy(edges, n, max_time=40.0)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
