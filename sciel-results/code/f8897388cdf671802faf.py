"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_paths(edges):
        if not edges:
            return []
        
        # Build adjacency list with edge tracking
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
            # Find the unused edge that is part of the longest possible simple path
            # Use a heuristic: pick an edge where both endpoints have high degree in unused subgraph
            # to maximize extension potential
            
            # Compute degrees in the unused-edge subgraph
            unused_deg = {}
            for u, v in unused_edges:
                unused_deg[u] = unused_deg.get(u, 0) + 1
                unused_deg[v] = unused_deg.get(v, 0) + 1
            
            # Pick the edge with the minimum sum of degrees (to avoid cutting off high-degree nodes)
            # Actually, let's pick an edge where at least one endpoint has degree 1 (leaf in unused subgraph)
            # This ensures we can extend from the other side
            best_edge = None
            best_score = float('inf')
            for e in unused_edges:
                u, v = e
                score = min(unused_deg.get(u, 0), unused_deg.get(v, 0))
                if score < best_score:
                    best_score = score
                    best_edge = e
                if best_score == 0:
                    break
            
            u, v = best_edge
            unused_edges.remove(best_edge)
            
            path = [u, v]
            visited_in_path = {u, v}
            
            # Greedily extend from both ends, preferring to continue in the direction
            # that has more options
            while True:
                # Try to extend from the start
                can_extend_start = False
                current = path[0]
                for neighbor in adj.get(current, []):
                    edge = (current, neighbor) if current < neighbor else (neighbor, current)
                    if edge in unused_edges and neighbor not in visited_in_path:
                        unused_edges.remove(edge)
                        path.insert(0, neighbor)
                        visited_in_path.add(neighbor)
                        can_extend_start = True
                        break
                
                if can_extend_start:
                    continue
                
                # Try to extend from the end
                can_extend_end = False
                current = path[-1]
                for neighbor in adj.get(current, []):
                    edge = (current, neighbor) if current < neighbor else (neighbor, current)
                    if edge in unused_edges and neighbor not in visited_in_path:
                        unused_edges.remove(edge)
                        path.append(neighbor)
                        visited_in_path.add(neighbor)
                        can_extend_end = True
                        break
                
                if not can_extend_end:
                    break
            
            paths.append(path)
            
        return paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose_paths(edges)
        result.append(paths)
        
    return result
# EVOLVE-BLOCK-END
