import random
import time

def construct():
    n = 16
    possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_3_colorable(edges):
        # Map vertex to index 0..15
        # Use bitmasks for colors
        # Backtracking
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            # Try colors 0, 1, 2
            # Optimization: try colors that are not used by neighbors
            used = 0
            for u in range(n):
                if adj[v] & (1 << u):
                    if colors[u] != -1:
                        used |= (1 << colors[u])
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
            colors[v] = -1
            return False
        
        return backtrack(0)

    best_edges = []
    best_count = -1
    start_time = time.time()
    
    # Fixed seed for determinism
    rng = random.Random(42)
    
    # We can run for a certain number of iterations or time limit
    # 120s limit. Let's say 50s for search.
    
    while time.time() - start_time < 60:
        # Generate random maximal triangle-free graph
        edges = []
        adj = [0] * n
        # Shuffle edges
        shuffled = possible_edges[:]
        rng.shuffle(shuffled)
        
        for u, v in shuffled:
            # Check if u and v have common neighbor
            # If adj[u] & adj[v] != 0, then adding (u,v) creates a triangle
            if not (adj[u] & adj[v]):
                edges.append([u, v])
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Check 3-colorability
        if not is_3_colorable(edges):
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges[:]
                # Optional: if we hit a very high number, we might break?
                # Max possible is 64. If we get > 40, that's great.
                if best_count > 40:
                    # Unlikely to get much higher, but keep searching a bit more?
                    pass

    # Fallback to Groetzsch if no better found (though we expect to find something)
    if best_count < 20:
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]
        
    return best_edges
