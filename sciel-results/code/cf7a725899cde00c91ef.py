"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_simple_paths(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = {}
        for u, v in edges:
            if u not in adj:
                adj[u] = []
            if v not in adj:
                adj[v] = []
            adj[u].append(v)
            adj[v].append(u)
        
        # Set of unused edges for O(1) lookup
        unused_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        
        paths = []
        
        while unused_edges:
            # Pick an arbitrary unused edge to start a new path
            start_edge = next(iter(unused_edges))
            u, v = start_edge
            unused_edges.remove(start_edge)
            
            path = [u, v]
            visited_in_path = {u, v}
            
            # Extend from the start (u)
            while True:
                current = path[0]
                found_next = False
                for neighbor in adj.get(current, []):
                    edge = (current, neighbor) if current < neighbor else (neighbor, current)
                    if edge in unused_edges and neighbor not in visited_in_path:
                        # Extend
                        unused_edges.remove(edge)
                        path.insert(0, neighbor)
                        visited_in_path.add(neighbor)
                        found_next = True
                        break
                if not found_next:
                    break
            
            # Extend from the end (v)
            while True:
                current = path[-1]
                found_next = False
                for neighbor in adj.get(current, []):
                    edge = (current, neighbor) if current < neighbor else (neighbor, current)
                    if edge in unused_edges and neighbor not in visited_in_path:
                        # Extend
                        unused_edges.remove(edge)
                        path.append(neighbor)
                        visited_in_path.add(neighbor)
                        found_next = True
                        break
                if not found_next:
                    break
            
            paths.append(path)
            
        return paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose_simple_paths(edges)
        result.append(paths)
        
    return result
# EVOLVE-BLOCK-END
