"""Rewritten program: decompose fixed instance graphs into few cycles + edges."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_longest_cycle(remaining, adj, n, rng, max_depth=60):
    """Find a cycle in the remaining graph using randomized DFS. Returns list of vertices or None."""
    if not remaining:
        return None
    
    # Pick a random starting vertex with degree >= 2
    candidates = [u for u in adj if len(adj[u]) >= 2]
    if not candidates:
        return None
    
    start = rng.choice(candidates)
    
    # Try multiple random walks to find a cycle
    for attempt in range(20):
        visited = set()
        path = []
        parent = {}
        
        def dfs(u, depth):
            if depth > max_depth:
                return False
            visited.add(u)
            path.append(u)
            
            neighbors = list(adj[u])
            rng.shuffle(neighbors)
            
            for v in neighbors:
                e = (min(u, v), max(u, v))
                if e not in remaining:
                    continue
                if v == start and len(path) >= 3:
                    return True
                if v not in visited:
                    parent[v] = u
                    if dfs(v, depth + 1):
                        return True
            
            path.pop()
            visited.discard(u)
            return False
        
        if dfs(start, 0):
            # Extract cycle from path
            # The cycle is from start to the last occurrence of start in path
            # Actually, we need to find where start appears and trace back
            # Since we're doing DFS and found v == start, the path contains the cycle
            # Find the index of start in path
            idx = path.index(start)
            cycle = path[idx:]
            return cycle
    
    return None

def _decompose(edges, n, seed, time_limit=35.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []
    
    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    
    for restart in range(2000):
        if time.time() - start_time > time_limit:
            break
        
        remaining = set(edges)
        pieces = []
        
        # Build adjacency
        adj = defaultdict(list)
        for u, v in remaining:
            adj[u].append(v)
            adj[v].append(u)
        
        rng = random.Random(seed * 1000 + restart)
        
        while remaining:
            # Try to find a cycle
            found_cycle = _find_longest_cycle(remaining, adj, n, rng)
            
            if found_cycle:
                # Extract the cycle edges
                cycle_edges = set()
                for i in range(len(found_cycle)):
                    u = found_cycle[i]
                    v = found_cycle[(i + 1) % len(found_cycle)]
                    e = (min(u, v), max(u, v))
                    cycle_edges.add(e)
                
                # Verify all cycle edges are in remaining
                valid = all(e in remaining for e in cycle_edges)
                if valid:
                    # Remove cycle edges from remaining
                    for e in cycle_edges:
                        remaining.discard(e)
                        u, v = e
                        if v in adj[u]:
                            adj[u].remove(v)
                        if u in adj[v]:
                            adj[v].remove(u)
                    pieces.append(found_cycle)
                    continue
            
            # No cycle found; take a single edge
            # Pick an edge incident to a high-degree vertex for better future cycles
            best_edge = None
            best_deg = -1
            for u in list(adj.keys()):
                if adj[u]:
                    deg = len(adj[u])
                    if deg > best_deg:
                        best_deg = deg
                        best_edge = u
            
            if best_edge is not None and adj[best_edge]:
                v = rng.choice(adj[best_edge])
                e = (min(best_edge, v), max(best_edge, v))
                if e in remaining:
                    remaining.discard(e)
                    if v in adj[best_edge]:
                        adj[best_edge].remove(v)
                    if best_edge in adj[v]:
                        adj[v].remove(best_edge)
                    pieces.append([best_edge, v])
                else:
                    # Fallback: take any remaining edge
                    e = next(iter(remaining))
                    remaining.discard(e)
                    u, v = e
                    if v in adj[u]:
                        adj[u].remove(v)
                    if u in adj[v]:
                        adj[v].remove(u)
                    pieces.append([u, v])
            else:
                # Fallback: take any remaining edge
                e = next(iter(remaining))
                remaining.discard(e)
                u, v = e
                if v in adj[u]:
                    adj[u].remove(v)
                if u in adj[v]:
                    adj[v].remove(u)
                pieces.append([u, v])
        
        current_count = len(pieces)
        if current_count < best_count:
            best_count = current_count
            best_pieces = [list(p) for p in pieces]
    
    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]
    
    return best_pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, seed, time_limit=40.0)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
