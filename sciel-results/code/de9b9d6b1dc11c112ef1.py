"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from random import Random

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
        Find a minimum number of edge-disjoint paths that cover all edges.
        
        The minimum number of paths in an edge path cover is:
        max(1, (number_of_odd_degree_vertices + 1) // 2) if there are odd vertices,
        else 1 if there are edges, else 0.
        
        Wait, that's not quite right. Let me think more carefully.
        
        Actually, the minimum number of trails (not necessarily simple paths) that cover all edges
        in a connected graph is max(1, odd_vertices / 2) where odd_vertices is the number of vertices
        with odd degree. But we need simple paths, not trails.
        
        However, for a connected graph, we can always decompose the edges into exactly
        max(1, odd_vertices // 2) trails. To make them simple paths, we might need to split
        trails at repeated vertices.
        
        A better approach: use the fact that any connected graph can be decomposed into
        at most (odd_vertices / 2) + 1 simple paths? No, that's not right either.
        
        Let me use a different approach. The key insight is:
        - A connected graph with 2k odd-degree vertices can be decomposed into k trails
        - Each trail can be split into simple paths
        
        Actually, let's just use a practical approach:
        1. Find all odd-degree vertices
        2. Pair them up greedily
        3. Add "virtual" edges between paired odd vertices to make all degrees even
        4. Find an Eulerian circuit in the augmented graph
        5. Remove the virtual edges to split the circuit into paths
        
        This gives us exactly k = odd_count / 2 paths (or 1 if no odd vertices and edges exist).
        
        But we need the paths to be simple (no repeated vertices). The Eulerian approach
        gives trails, not necessarily simple paths.
        
        For a simpler but still effective approach, let's use:
        - Hierholzer's algorithm to find trails
        - Then split trails at repeated vertices to get simple paths
        
        Actually, for the purpose of this task, let's use a greedy approach:
        1. Start with all edges uncovered
        2. Repeatedly find a longest possible simple path using uncovered edges
        3. Remove those edges and repeat
        
        This is O(E^2) but should be fast enough for small graphs.
        
        Let me implement a cleaner version using the pairing of odd vertices.
        """
        if not edges:
            return []
        
        # Build adjacency and degree
        adj = {}
        degree = {}
        edge_set = set()
        for u, v in edges:
            if u not in adj:
                adj[u] = []
            if v not in adj:
                adj[v] = []
            adj[u].append(v)
            adj[v].append(u)
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
            edge_set.add((u, v))
        
        # Find odd-degree vertices
        odd_vertices = [v for v in degree if degree[v] % 2 == 1]
        
        # The minimum number of trails needed is max(1, len(odd_vertices) // 2)
        # We'll use a greedy approach to find simple paths
        
        # Use a recursive backtracking / greedy approach to find paths
        # For small graphs, we can try to find good paths greedily
        
        remaining_edges = set(edge_set)
        paths = []
        
        def find_path(start, remaining):
            """Find a simple path starting from 'start' using only edges in 'remaining'."""
            if not remaining:
                return None
            
            # Find neighbors of start via remaining edges
            neighbors = []
            for (u, v) in remaining:
                if u == start:
                    neighbors.append(v)
                elif v == start:
                    neighbors.append(u)
            
            if not neighbors:
                return None
            
            best_path = None
            for nb in neighbors:
                # Try to extend the path from nb
                new_remaining = set(remaining)
                # Remove the edge (start, nb)
                if (start, nb) in new_remaining:
                    new_remaining.remove((start, nb))
                elif (nb, start) in new_remaining:
                    new_remaining.remove((nb, start))
                
                # Find longest path from nb
                path = [start, nb]
                visited = {start, nb}
                current = nb
                while True:
                    extended = False
                    for (u, v) in new_remaining:
                        if u == current and v not in visited:
                            new_remaining.remove((u, v))
                            path.append(v)
                            visited.add(v)
                            current = v
                            extended = True
                            break
                        elif v == current and u not in visited:
                            new_remaining.remove((u, v))
                            path.append(u)
                            visited.add(u)
                            current = u
                            extended = True
                            break
                    if not extended:
                        break
                
                if best_path is None or len(path) > len(best_path):
                    best_path = path
            
            return best_path
        
        # Greedy: repeatedly find a path
        while remaining_edges:
            # Pick a start vertex: prefer odd-degree vertices in remaining graph
            rem_degree = {}
            for (u, v) in remaining_edges:
                rem_degree[u] = rem_degree.get(u, 0) + 1
                rem_degree[v] = rem_degree.get(v, 0) + 1
            
            # Find vertices with odd degree in remaining graph
            rem_odd = [v for v in rem_degree if rem_degree[v] % 2 == 1]
            
            if rem_odd:
                start = rem_odd[0]
            else:
                start = next(iter(rem_degree))
            
            path = find_path(start, remaining_edges)
            if path is None:
                # Should not happen, but fallback
                break
            
            # Remove edges from path
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if (u, v) in remaining_edges:
                    remaining_edges.remove((u, v))
                elif (v, u) in remaining_edges:
                    remaining_edges.remove((v, u))
            
            paths.append(path)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = min_path_cover(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
