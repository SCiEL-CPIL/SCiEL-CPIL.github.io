"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def _decompose(edges, n, seed, time_limit=10.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []
    
    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    
    for restart in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Create a fresh copy of edges for this restart
        remaining = set(edges)
        pieces = []
        
        # Build adjacency
        adj = defaultdict(list)
        for u, v in remaining:
            adj[u].append(v)
            adj[v].append(u)
        
        # Greedy cycle extraction with randomized tie-breaking
        rng = random.Random(seed + restart)
        
        while remaining:
            # Find a vertex with highest degree to start from
            # Randomize among top-degree vertices for diversity
            candidates = [(len(adj[u]), u) for u in adj if adj[u]]
            if not candidates:
                break
            
            # Pick from top 5 candidates randomly
            candidates.sort(reverse=True)
            top_k = min(5, len(candidates))
            _, start = rng.choice(candidates[:top_k])
            
            # Try to find a cycle starting from start using DFS
            found_cycle = None
            visited = set()
            path = []
            
            def dfs(u, depth):
                nonlocal found_cycle, visited, path
                if found_cycle is not None:
                    return True
                if depth > n + 5:
                    return False
                
                visited.add(u)
                path.append(u)
                
                # Get neighbors in remaining edges
                neighbors = []
                for v in adj[u]:
                    e1 = (min(u, v), max(u, v))
                    e2 = (min(v, u), max(v, u))
                    if e1 in remaining or e2 in remaining:
                        neighbors.append(v)
                
                # Shuffle neighbors for randomization
                rng.shuffle(neighbors)
                
                for v in neighbors:
                    if v == start and len(path) >= 3:
                        # Found a cycle
                        found_cycle = list(path)
                        return True
                    if v not in visited:
                        if dfs(v, depth + 1):
                            return True
                
                path.pop()
                visited.discard(u)
                return False
            
            dfs(start, 0)
            
            if found_cycle:
                # Extract the cycle edges
                cycle_edges = set()
                for i in range(len(found_cycle)):
                    u = found_cycle[i]
                    v = found_cycle[(i + 1) % len(found_cycle)]
                    e = (min(u, v), max(u, v))
                    cycle_edges.add(e)
                
                # Remove cycle edges from remaining
                for e in cycle_edges:
                    if e in remaining:
                        remaining.discard(e)
                        u, v = e
                        adj[u].remove(v)
                        adj[v].remove(u)
                
                pieces.append(found_cycle)
            else:
                # No cycle found; take a single edge
                # Pick an edge incident to a high-degree vertex
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
                        adj[best_edge].remove(v)
                        adj[v].remove(best_edge)
                        pieces.append([best_edge, v])
                else:
                    # Fallback: take any remaining edge
                    e = next(iter(remaining))
                    remaining.discard(e)
                    u, v = e
                    adj[u].remove(v)
                    adj[v].remove(u)
                    pieces.append([u, v])
        
        # Count pieces
        current_count = len(pieces)
        if current_count < best_count:
            best_count = current_count
            best_pieces = pieces
    
    if best_pieces is None:
        # Fallback: all edges as single pieces
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
