"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose edge set into minimum number of simple edge-disjoint paths."""
        if not edges:
            return []

        # Build adjacency from remaining edges
        def build_adj(remaining):
            adj = defaultdict(list)
            for u, v in remaining:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        def find_longest_path(remaining, adj, max_depth=500):
            """Find a long simple path in the remaining graph using DFS with pruning."""
            best_path = []
            visited = set()
            
            # Try from each vertex as start
            vertices = sorted(remaining[0][0] if remaining else [0])
            for u, v in remaining:
                if u not in vertices:
                    vertices.append(u)
                if v not in vertices:
                    vertices.append(v)
            
            for start in vertices:
                if time.time() > DEADLINE - 5:
                    break
                # DFS to find longest path starting from 'start'
                path = [start]
                used_edges = set()
                
                def dfs(current, path, used_edges):
                    nonlocal best_path
                    if len(path) > len(best_path):
                        best_path = path[:]
                    
                    # Try to extend
                    for nb in adj.get(current, []):
                        edge = tuple(sorted((current, nb)))
                        if edge in used_edges:
                            continue
                        if nb in path:  # must be simple
                            continue
                        path.append(nb)
                        used_edges.add(edge)
                        dfs(nb, path, used_edges)
                        path.pop()
                        used_edges.discard(edge)
                
                dfs(start, path, used_edges)
            
            return best_path

        remaining = set(edges)
        paths = []
        
        while remaining:
            if time.time() > DEADLINE - 10:
                # Fallback: each remaining edge as its own path
                for u, v in remaining:
                    paths.append([u, v])
                break
            
            adj = build_adj(remaining)
            path = find_longest_path(remaining, adj)
            
            if not path or len(path) < 2:
                # Shouldn't happen if remaining is non-empty, but safety
                for u, v in remaining:
                    paths.append([u, v])
                break
            
            # Remove edges used in this path
            for i in range(len(path) - 1):
                edge = tuple(sorted((path[i], path[i+1])))
                remaining.discard(edge)
            
            paths.append(path)
        
        return paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE - 20:
            # Not enough time, use simple fallback for remaining instances
            for remaining_spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(result):]:
                edges = instance(*remaining_spec)
                result.append([[u, v] for u, v in edges])
            break
        
        edges = instance(*spec)
        paths = decompose(edges)
        result.append(paths)

    return result
# EVOLVE-BLOCK-END
