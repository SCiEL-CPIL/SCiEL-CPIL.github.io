import sys
sys.setrecursionlimit(10000)

def construct():
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_greedy_cycle(adj, n, start, rng):
        """Find a simple cycle using a greedy path extension strategy.
        Returns list of vertices or None."""
        if not adj[start]:
            return None
        
        path = [start]
        visited = {start}
        last = start
        
        # Heuristic: try to extend the path as long as possible
        while True:
            # Get unvisited neighbors
            neighbors = [v for v in adj[last] if v not in visited]
            if not neighbors:
                break
            
            # Choose a neighbor. 
            # Strategy: Randomly pick, but maybe prefer those with higher degree to keep path alive?
            # Or lower degree to avoid getting stuck? 
            # Let's stick to random for now, with many retries.
            next_v = rng.choice(neighbors)
            path.append(next_v)
            visited.add(next_v)
            last = next_v
            
            # Optional: If we hit a dead end and path is short, we could backtrack, 
            # but simple greedy is fast.
        
        # Check if we can close the cycle
        if len(path) >= 3 and start in adj[last]:
            return path
        return None

    def find_dfs_cycle(adj, n, start, rng):
        """Fallback DFS cycle finder."""
        if not adj[start]:
            return None
        
        visited = [False] * n
        path = []
        
        def dfs(u):
            visited[u] = True
            path.append(u)
            neighbors = list(adj[u])
            rng.shuffle(neighbors)
            for v in neighbors:
                if v == path[0] and len(path) >= 3:
                    return path[:]
                if not visited[v]:
                    result = dfs(v)
                    if result:
                        return result
            path.pop()
            return None
        
        result = dfs(start)
        return result

    def decompose(edges):
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
        
        pieces = []
        rng = Random(42) # Fixed seed for heuristic stability
        
        while edge_set:
            active_vertices = [v for v in range(n) if adj[v]]
            if not active_vertices:
                break
                
            # Try multiple random starts to find a long cycle
            best_cycle = None
            # Use a mix of greedy and DFS? 
            # Greedy is faster. Let's do more greedy trials.
            for _ in range(200):
                # Prioritize high degree vertices?
                # Sort active vertices by degree descending? 
                # Just random choice is okay, but high degree is better for long cycles.
                # Let's pick a random vertex, but if we want to be smarter:
                # start = max(active_vertices, key=lambda x: len(adj[x]))
                # But random is better for diversity.
                start = rng.choice(active_vertices)
                
                # Try greedy first
                cycle = find_greedy_cycle(adj, n, start, rng)
                
                # If greedy fails or gives short cycle, try DFS?
                # DFS is slower. Let's just do greedy.
                # If greedy returns None, it means the path didn't close.
                
                if cycle:
                    if best_cycle is None or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                        if len(best_cycle) == n: # Hamiltonian
                            break
            
            if best_cycle is None:
                # If greedy failed to find any cycle, fall back to DFS for a few vertices
                # to ensure we don't miss cycles in small components
                for v in active_vertices:
                    cycle = find_dfs_cycle(adj, n, v, rng)
                    if cycle:
                        if best_cycle is None or len(cycle) > len(best_cycle):
                            best_cycle = cycle
                        if len(best_cycle) == n:
                            break
                
            if best_cycle is None:
                # No cycles found, add remaining edges
                for u, v in sorted(edge_set):
                    pieces.append([u, v])
                break
            
            # Remove cycle edges
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                edge = (min(u, v), max(u, v))
                if edge in edge_set:
                    edge_set.remove(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            
            pieces.append(best_cycle[:])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
