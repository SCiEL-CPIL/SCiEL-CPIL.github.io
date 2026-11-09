import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 145.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1
        adj = [set() for _ in range(n)]
        deg = [0] * n
        
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        paths = []
        
        # Keep track of remaining edges count to avoid O(N) check in while loop
        remaining_edges = len(edges)
        
        while remaining_edges > 0:
            # Find a vertex with degree > 0
            u = -1
            for i in range(n):
                if deg[i] > 0:
                    u = i
                    break
            
            if u == -1:
                break # Should not happen if remaining_edges > 0
            
            path = [u]
            current = u
            
            # Extend path
            while deg[current] > 0:
                # Pick a neighbor
                # To improve path quality, we can pick the neighbor with highest degree (Watt's heuristic)
                # or just any neighbor.
                best_next = -1
                max_deg_neighbor = -1
                for v in adj[current]:
                    if deg[v] > max_deg_neighbor:
                        max_deg_neighbor = deg[v]
                        best_next = v
                
                if best_next == -1:
                    break # Safety, should not happen if deg[current] > 0
                
                # Move to best_next
                adj[current].remove(best_next)
                adj[best_next].remove(current)
                deg[current] -= 1
                deg[best_next] -= 1
                remaining_edges -= 1
                
                path.append(best_next)
                current = best_next
            
            # A path must have at least 2 vertices.
            # If len(path) == 1, it means we couldn't extend, which implies deg[u] was 0 initially.
            # But we selected u with deg[u] > 0.
            # So len(path) should always be >= 2.
            if len(path) >= 2:
                paths.append(path)
            else:
                # Fallback: if somehow a single vertex path is generated, 
                # we must handle it. But this indicates a logic error.
                # For now, we skip it, but this would leave edges uncovered.
                # However, given deg[u] > 0, this branch should be dead code.
                pass

        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
        if time.time() > deadline:
            # Fallback if too slow (unlikely)
            pass

    return results
