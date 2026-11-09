import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        # Build adjacency and degree structures
        # adj[u] = set of neighbors
        # deg[u] = number of edges
        n = max(max(e) for e in edges) + 1 if edges else 0
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        paths = []
        
        while any(deg):
            if time.time() > deadline:
                # If we exceed deadline, we return what we have. 
                # However, the greedy loop is very fast for N=30, 
                # so this is just a safety net.
                # To ensure validity, we must cover ALL edges. 
                # If we cut off, we should append remaining edges as single-edge paths.
                # But since it's fast, we assume we finish.
                pass

            # Find vertex with max degree
            max_deg = max(deg)
            if max_deg == 0:
                break
            
            # Find the vertex u with max degree
            u = deg.index(max_deg)
            
            path = [u]
            deg[u] -= 1 # Conceptually we start here, but we remove edges as we traverse
            
            # We need to track used edges to ensure we don't reuse them.
            # Actually, the `adj` and `deg` updates handle this.
            # When we traverse (u, v), we remove edge (u,v).
            
            # Let's restart the logic slightly to be cleaner:
            # We don't pre-decrement. We pick u.
            # Path starts at u.
            # We need to enter a loop to extend.
            
            # Wait, the above `deg[u] -= 1` was premature. 
            # Let's do it properly:
            path = [u]
            current = u
            
            while True:
                # Find next vertex
                best_next = -1
                best_deg = -1
                for v in adj[current]:
                    if deg[v] > best_deg:
                        best_deg = deg[v]
                        best_next = v
                
                if best_next == -1:
                    break
                
                # Move to best_next
                # Remove edge (current, best_next)
                adj[current].remove(best_next)
                adj[best_next].remove(current)
                deg[current] -= 1
                deg[best_next] -= 1
                
                path.append(best_next)
                current = best_next
                
                # If current has no more edges, stop extending this path
                if deg[current] == 0:
                    break
            
            paths.append(path)
            
        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
        # Check time
        if time.time() > deadline:
             # Fallback: if we run out of time, we can't easily "finish" 
             # but the algorithm is O(E^2) which is tiny.
             pass

    return results

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

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
        
        # Loop until no edges remain
        while any(deg):
            # Find vertex with max degree to start a new path
            # This is a heuristic: starting at a high-degree vertex 
            # allows us to potentially form a long path.
            max_deg = max(deg)
            if max_deg == 0:
                break
            
            u = deg.index(max_deg)
            path = [u]
            current = u
            
            # Extend the path greedily
            while deg[current] > 0:
                # Choose neighbor with highest degree
                best_next = -1
                best_neighbor_deg = -1
                for v in adj[current]:
                    if deg[v] > best_neighbor_deg:
                        best_neighbor_deg = deg[v]
                        best_next = v
                
                if best_next == -1:
                    break
                
                # Traverse edge (current, best_next)
                adj[current].remove(best_next)
                adj[best_next].remove(current)
                deg[current] -= 1
                deg[best_next] -= 1
                
                path.append(best_next)
                current = best_next
                
            paths.append(path)
            
            # Check time
            if time.time() > deadline:
                 # In case of extreme slowness (unlikely), we could break,
                 # but we need to return a valid decomposition.
                 # Since we modify state destructively, if we break here,
                 # remaining edges are not covered.
                 # However, for N=30, this loop runs in < 1ms.
                 pass

        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)

    return results
# EVOLVE-BLOCK-END
