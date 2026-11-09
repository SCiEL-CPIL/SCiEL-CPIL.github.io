"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose_graph(edges, time_limit=45):
        """
        Decompose graph into edge-disjoint paths covering all edges.
        Strategy: 
        1. Build adjacency.
        2. Repeatedly find a long simple path in the remaining graph.
        3. Remove those edges.
        4. Try to merge paths if possible (optional, complex).
        
        We use a heuristic: start from a high-degree node, greedily extend 
        the path to a simple path, prioritizing neighbors that keep options open.
        """
        start_time = time.time()
        
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        def find_long_path():
            """Find a simple path in the remaining graph."""
            if not remaining_edges:
                return None
            
            # Choose start node: highest degree in remaining graph
            degrees = {u: len(adj[u] & set(n for e in remaining_edges for n in (e[0], e[1]))) for u in adj}
            # Simpler: just count remaining edges incident to each node
            rem_deg = defaultdict(int)
            for u, v in remaining_edges:
                rem_deg[u] += 1
                rem_deg[v] += 1
            
            if not rem_deg:
                return None
                
            start = max(rem_deg.keys(), key=lambda x: (rem_deg[x], -x))
            
            # Greedy path extension with backtracking for small improvements
            # Since graphs are sparse and small, we can do a DFS to find a longer path
            # But for speed, we do a heuristic greedy first.
            
            visited = {start}
            path = [start]
            current = start
            
            while True:
                # Get neighbors of current in remaining graph
                neighbors = [n for n in adj[current] if (min(current, n), max(current, n)) in remaining_edges]
                if not neighbors:
                    break
                
                # Choose next neighbor
                # Heuristic: choose neighbor with lowest remaining degree (to avoid getting stuck)
                # or highest? Usually lowest to "consume" dead-ends first? 
                # Let's try: choose neighbor that has fewest options (to prevent creating isolated edges later)
                # Actually, for path cover, we want to use edges that are "critical".
                # Let's try a simple heuristic: pick neighbor with min remaining degree.
                next_node = min(neighbors, key=lambda n: rem_deg.get(n, 0))
                
                # Remove edge
                e = (min(current, next_node), max(current, next_node))
                remaining_edges.remove(e)
                rem_deg[current] -= 1
                rem_deg[next_node] -= 1
                
                # If next_node is already in path, we can't go there (simple path)
                if next_node in visited:
                    # This shouldn't happen if we check, but let's be safe
                    # Restore
                    remaining_edges.add(e)
                    rem_deg[current] += 1
                    rem_deg[next_node] += 1
                    # Try another neighbor
                    neighbors.remove(next_node)
                    if not neighbors:
                        break
                    next_node = min(neighbors, key=lambda n: rem_deg.get(n, 0))
                    e = (min(current, next_node), max(current, next_node))
                    remaining_edges.remove(e)
                    rem_deg[current] -= 1
                    rem_deg[next_node] -= 1
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            return path
        
        # Main loop
        while remaining_edges:
            if time.time() - start_time > time_limit:
                # Fallback: add remaining edges as individual paths
                for u, v in remaining_edges:
                    paths.append([u, v])
                break
            
            path = find_long_path()
            if path is None:
                # Should not happen if remaining_edges not empty, but safety
                for u, v in remaining_edges:
                    paths.append([u, v])
                break
            if len(path) > 1:
                paths.append(path)
            else:
                # Single node? Should not happen if edges remain
                break
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Emergency: return trivial decomposition for remaining
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        paths = decompose_graph(edges, time_limit=40)
        results.append(paths)
        
        # Optional: Try to improve by merging paths?
        # For now, just return what we have.
        
    return results
# EVOLVE-BLOCK-END
