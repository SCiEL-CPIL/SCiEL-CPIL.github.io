"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 130  # safety margin under 150s wall-clock
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose_graph(edges):
        """Decompose a graph into cycles and edges, minimizing the number of pieces."""
        if not edges:
            return []
        
        # Build adjacency structure
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Track remaining edges
        remaining = set(edges)
        pieces = []
        
        # Greedy: repeatedly find a long cycle in the remaining graph
        # Use DFS with backtracking to find a simple cycle
        # Prioritize starting from high-degree vertices
        
        while remaining:
            # Find vertex with highest degree in remaining graph
            deg = {}
            for u, v in remaining:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            # Sort vertices by degree descending
            vertices_sorted = sorted(deg.keys(), key=lambda x: -deg[x])
            
            found_cycle = None
            
            # Try to find a cycle starting from each high-degree vertex
            for start in vertices_sorted:
                if start not in deg or deg[start] < 2:
                    continue
                
                # DFS to find a cycle back to start
                # Limit search depth to avoid explosion
                max_depth = 30
                found = _find_cycle_dfs(start, start, remaining, max_depth)
                if found and found_cycle is None:
                    found_cycle = found
                elif found and found_cycle is not None and len(found) > len(found_cycle):
                    found_cycle = found
            
            if found_cycle:
                # Remove the cycle edges from remaining
                for i in range(len(found_cycle)):
                    u = found_cycle[i]
                    v = found_cycle[(i + 1) % len(found_cycle)]
                    e = tuple(sorted((u, v)))
                    remaining.discard(e)
                pieces.append(found_cycle)
            else:
                # No cycle found; take a single edge
                e = next(iter(remaining))
                remaining.discard(e)
                pieces.append([e[0], e[1]])
        
        return pieces
    
    def _find_cycle_dfs(start, current, remaining, max_depth):
        """Find a simple cycle from start back to start using DFS."""
        path = [start]
        visited = {start}
        
        def dfs(node, depth):
            if depth >= max_depth:
                return None
            for neighbor in sorted(remaining_neighbors(node, remaining)):
                if neighbor == start and len(path) >= 3:
                    return list(path)
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    result = dfs(neighbor, depth + 1)
                    if result:
                        return result
                    path.pop()
                    visited.discard(neighbor)
            return None
        
        return dfs(start, 0)
    
    def remaining_neighbors(node, remaining):
        """Get neighbors of node in the remaining edge set."""
        neighbors = set()
        for u, v in remaining:
            if u == node:
                neighbors.add(v)
            elif v == node:
                neighbors.add(u)
        return neighbors
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Fallback: just return edges
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        pieces = decompose_graph(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
