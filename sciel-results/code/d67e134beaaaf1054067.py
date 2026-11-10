"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    from random import Random
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        used_edges = set()
        
        # Greedy: repeatedly find a cycle in the remaining graph
        def find_cycle(adj, used_edges):
            """Find a cycle in the current graph using DFS."""
            visited = set()
            
            def dfs(start, path, path_set):
                for neighbor in sorted(adj[start]):
                    edge = tuple(sorted((start, neighbor)))
                    if edge in used_edges:
                        continue
                    if neighbor == start and len(path) >= 3:
                        return path + [start]
                    if neighbor not in path_set:
                        path_set.add(neighbor)
                        path.append(neighbor)
                        result = dfs(neighbor, path, path_set)
                        if result:
                            return result
                        path.pop()
                        path_set.discard(neighbor)
                return None
            
            for start in sorted(adj.keys()):
                if start in visited:
                    continue
                visited.add(start)
                result = dfs(start, [start], {start})
                if result:
                    return result
            return None
        
        # Phase 1: Greedy cycle extraction
        while True:
            if time.time() > deadline:
                break
            cycle = find_cycle(adj, used_edges)
            if not cycle:
                break
            # Mark cycle edges as used
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i + 1]
                edge = tuple(sorted((u, v)))
                if edge in edge_set:
                    used_edges.add(edge)
            pieces.append(cycle)
        
        # Phase 2: Local search - try to convert single edges into cycles
        # Find remaining single edges
        remaining = [e for e in edge_set if e not in used_edges]
        
        # Try to find cycles among remaining edges by adding back some edges
        # This is complex, so we'll do a simpler local search:
        # Try swapping pieces to see if we can reduce count
        
        # For now, just return what we have
        result = pieces.copy()
        for e in remaining:
            result.append(list(e))
        
        return result
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Return trivial solution if time is up
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
