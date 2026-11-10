"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from random import Random
from collections import defaultdict

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
        1. Compute degrees and find odd-degree vertices.
        2. The minimum number of paths is at least ceil(odd_count / 2).
        3. Use a greedy approach to find long simple paths, prioritizing odd-degree
           vertices as endpoints to minimize the total number of paths.
        4. After initial path extraction, try to merge paths if possible.
        """
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(list)
        degree = defaultdict(int)
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degree[u] += 1
            degree[v] += 1
            edge_set.add((min(u, v), max(u, v)))
        
        remaining_edges = set(edge_set)
        paths = []
        
        def find_longest_path(start, remaining):
            """
            Find the longest simple path starting from 'start' using DFS with backtracking.
            Uses heuristic: always try to extend to vertices with higher remaining degree.
            """
            if not remaining:
                return None
            
            # Get neighbors of start via remaining edges
            def get_neighbors(vertex, rem_edges):
                neighbors = []
                for (u, v) in rem_edges:
                    if u == vertex:
                        neighbors.append(v)
                    elif v == vertex:
                        neighbors.append(u)
                return neighbors
            
            start_neighbors = get_neighbors(start, remaining)
            if not start_neighbors:
                return None
            
            best_path = None
            best_len = 0
            
            # Sort neighbors by remaining degree (descending) for better heuristic
            def rem_degree(v):
                deg = 0
                for (u, w) in remaining:
                    if u == v or w == v:
                        deg += 1
                return deg
            
            sorted_neighbors = sorted(start_neighbors, key=lambda x: rem_degree(x), reverse=True)
            
            for nb in sorted_neighbors:
                new_remaining = set(remaining)
                # Remove edge (start, nb)
                e1 = (min(start, nb), max(start, nb))
                if e1 in new_remaining:
                    new_remaining.remove(e1)
                
                path = [start, nb]
                visited = {start, nb}
                current = nb
                
                # Extend the path greedily
                while True:
                    curr_neighbors = get_neighbors(current, new_remaining)
                    # Filter to unvisited vertices
                    valid_neighbors = [v for v in curr_neighbors if v not in visited]
                    if not valid_neighbors:
                        break
                    
                    # Choose neighbor with highest remaining degree
                    valid_neighbors.sort(key=lambda x: rem_degree(x), reverse=True)
                    next_v = valid_neighbors[0]
                    
                    # Remove edge (current, next_v)
                    e2 = (min(current, next_v), max(current, next_v))
                    if e2 in new_remaining:
                        new_remaining.remove(e2)
                    
                    path.append(next_v)
                    visited.add(next_v)
                    current = next_v
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            return best_path
        
        # Greedy: repeatedly find a path
        while remaining_edges:
            # Compute degrees in remaining graph
            rem_degree = defaultdict(int)
            for (u, v) in remaining_edges:
                rem_degree[u] += 1
                rem_degree[v] += 1
            
            # Find vertices with odd degree in remaining graph
            rem_odd = [v for v in rem_degree if rem_degree[v] % 2 == 1]
            
            if rem_odd:
                # Start from an odd-degree vertex
                start = rem_odd[0]
            else:
                # All degrees even, start from any vertex with edges
                start = max(rem_degree.keys(), key=lambda x: rem_degree[x])
            
            path = find_longest_path(start, remaining_edges)
            if path is None:
                # Fallback: just take one edge
                e = next(iter(remaining_edges))
                path = [e[0], e[1]]
                remaining_edges.remove(e)
                paths.append(path)
                continue
            
            # Remove edges from path
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    remaining_edges.remove(e)
            
            paths.append(path)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = min_path_cover(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
