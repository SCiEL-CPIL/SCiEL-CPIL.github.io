"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from random import Random
from collections import deque

def construct():
    # Generate the instance graphs using the same method as the verifier
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def min_path_cover(edges):
        """
        Find a minimum number of edge-disjoint simple paths that cover all edges.
        Strategy:
        1. While edges remain:
           a. Compute degrees in the remaining graph.
           b. If there are odd-degree vertices, pick two (u, v) and find a simple path between them.
           c. If no odd-degree vertices, pick any edge (u, v) and find a simple path starting at u.
           d. Remove the edges of the path.
        """
        if not edges:
            return []
        
        # Build adjacency list and edge set for fast lookup
        adj = {}
        edge_set = set()
        for u, v in edges:
            if u not in adj:
                adj[u] = set()
            if v not in adj:
                adj[v] = set()
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        remaining_edges = set(edge_set)
        paths = []
        
        def get_degree(v):
            return sum(1 for (a, b) in remaining_edges if a == v or b == v)
        
        def find_simple_path(u, v, remaining):
            """
            Find a simple path between u and v using BFS.
            Returns list of vertices or None.
            """
            if u == v:
                return [u]
            
            # BFS
            queue = deque([(u, [u])])
            visited = {u}
            
            while queue:
                node, path = queue.popleft()
                if node == v:
                    return path
                
                # Get neighbors in remaining graph
                neighbors = []
                for (a, b) in remaining:
                    if a == node:
                        neighbors.append(b)
                    elif b == node:
                        neighbors.append(a)
                
                for nb in neighbors:
                    if nb not in visited:
                        # Check if edge (node, nb) is in remaining
                        if (node, nb) in remaining or (nb, node) in remaining:
                            visited.add(nb)
                            queue.append((nb, path + [nb]))
            return None
        
        def find_longest_simple_path(start, remaining):
            """
            Find a simple path starting at 'start' that is as long as possible.
            Uses a greedy DFS with backtracking for small graphs? 
            For now, use a greedy extension picking the neighbor with the most remaining options.
            """
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Get neighbors of current in remaining
                candidates = []
                for (a, b) in remaining:
                    if a == current:
                        candidates.append(b)
                    elif b == current:
                        candidates.append(a)
                
                if not candidates:
                    break
                
                # Heuristic: pick the candidate with the highest remaining degree
                best_nb = None
                best_deg = -1
                for nb in candidates:
                    deg = sum(1 for (a, b) in remaining if a == nb or b == nb)
                    if deg > best_deg:
                        best_deg = deg
                        best_nb = nb
                
                # Remove edge (current, best_nb)
                if (current, best_nb) in remaining:
                    remaining.remove((current, best_nb))
                elif (best_nb, current) in remaining:
                    remaining.remove((best_nb, current))
                
                path.append(best_nb)
                visited.add(best_nb)
                current = best_nb
                
            return path if len(path) > 1 else None

        while remaining_edges:
            # Compute degrees
            rem_degree = {}
            for (u, v) in remaining_edges:
                rem_degree[u] = rem_degree.get(u, 0) + 1
                rem_degree[v] = rem_degree.get(v, 0) + 1
            
            rem_odd = [v for v in rem_degree if rem_degree[v] % 2 == 1]
            
            if rem_odd:
                # Pick two odd vertices
                u = rem_odd[0]
                v = rem_odd[1] if len(rem_odd) > 1 else rem_odd[0]
                
                # Find path between u and v
                # We need to pass a copy of remaining to BFS? 
                # BFS doesn't modify remaining, so we can pass the set directly.
                path = find_simple_path(u, v, remaining_edges)
                
                if path is None:
                    # Fallback: pick any edge incident to u
                    # This should not happen if the graph is connected, 
                    # but if it's disconnected, we might need to handle components.
                    # For now, just pick a neighbor.
                    neighbors = []
                    for (a, b) in remaining_edges:
                        if a == u: neighbors.append(b)
                        elif b == u: neighbors.append(a)
                    if neighbors:
                        path = [u, neighbors[0]]
                    else:
                        break
            else:
                # All degrees even. Pick any edge (u, v)
                u, v = next(iter(remaining_edges))
                # Find a path starting at u
                # We want a long path to break the cycle effectively.
                # Let's use the greedy longest path finder.
                # Note: find_longest_simple_path modifies remaining, so we must be careful.
                # It removes edges as it goes. We need to ensure we only remove the edges 
                # of the final path.
                # Let's simulate or just use a non-destructive version?
                # The current find_longest_simple_path removes from 'remaining' which is passed by reference.
                # We should NOT remove from the global remaining_edges until we have the final path.
                # Let's create a local copy for the search? 
                # Or better, just use BFS for a path of length 1? No, that's bad.
                
                # Let's use a simple DFS to find a path.
                # Since we just need *a* path, and we want it long, 
                # let's try to extend greedily without modifying the global set first.
                
                path = [u]
                visited = {u}
                current = u
                local_used_edges = set()
                
                while True:
                    candidates = []
                    for (a, b) in remaining_edges:
                        if a == current and b not in visited:
                            candidates.append(b)
                        elif b == current and a not in visited:
                            candidates.append(a)
                    
                    if not candidates:
                        break
                    
                    # Pick candidate with max degree in remaining_edges
                    best_nb = None
                    best_deg = -1
                    for nb in candidates:
                        deg = rem_degree.get(nb, 0)
                        if deg > best_deg:
                            best_deg = deg
                            best_nb = nb
                    
                    path.append(best_nb)
                    visited.add(best_nb)
                    current = best_nb
                    
                if len(path) < 2:
                    # Should not happen if edges exist
                    break
            
            # Remove edges from the found path
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                if (a, b) in remaining_edges:
                    remaining_edges.remove((a, b))
                elif (b, a) in remaining_edges:
                    remaining_edges.remove((b, a))
            
            paths.append(path)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = min_path_cover(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
