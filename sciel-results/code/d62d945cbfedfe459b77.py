"""Decompose random graphs into few cycles and edges."""

def construct():
    import random
    from collections import defaultdict

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """
        Decompose an undirected graph into cycles and single edges.
        
        Strategy:
        1. Build adjacency list with edge tracking.
        2. While edges remain:
           a. Find a cycle using DFS with back-edge detection.
           b. Prefer longer cycles to minimize total piece count.
           c. If no cycle exists, take a single edge.
        
        Key improvement: Instead of just finding ANY cycle, we try to find
        a reasonably long cycle by doing a full DFS and collecting all cycles,
        then picking the longest one. This reduces the total number of pieces.
        """
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            # Find the longest cycle in the current graph
            best_cycle = None
            best_len = 0
            
            # Try DFS from each vertex with degree > 0
            for start in list(adj.keys()):
                if not adj[start]:
                    continue
                
                visited = set()
                parent = {}
                path = []
                cycles_found = []
                
                def dfs(u):
                    visited.add(u)
                    path.append(u)
                    for v in sorted(adj[u]):
                        if v == parent.get(u):
                            continue
                        if v in visited:
                            # Back edge found - extract cycle
                            if v in path:
                                idx = path.index(v)
                                cycle_vertices = path[idx:]
                                if len(cycle_vertices) >= 3:
                                    cycles_found.append(cycle_vertices)
                        else:
                            parent[v] = u
                            dfs(v)
                            # After returning from recursion, v is no longer on the current path
                            # but we keep it in visited to avoid revisiting
                    path.pop()
                
                dfs(start)
                
                # Find the longest cycle from this DFS
                for cyc in cycles_found:
                    if len(cyc) > best_len:
                        best_len = len(cyc)
                        best_cycle = cyc
                
                # Early termination: if we found a cycle that uses all remaining edges, stop
                if best_len == len(remaining_edges):
                    break
            
            if best_cycle is not None and best_len >= 3:
                # Verify the cycle edges are still in remaining_edges
                valid = True
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = tuple(sorted((u, v)))
                    if edge not in remaining_edges:
                        valid = False
                        break
                
                if valid:
                    # Remove the cycle edges
                    for i in range(len(best_cycle)):
                        u = best_cycle[i]
                        v = best_cycle[(i + 1) % len(best_cycle)]
                        edge = tuple(sorted((u, v)))
                        remaining_edges.discard(edge)
                        adj[u].discard(v)
                        adj[v].discard(u)
                    pieces.append(best_cycle)
                else:
                    # Fallback: find any cycle or take an edge
                    cycle = _find_any_cycle(adj, remaining_edges)
                    if cycle:
                        for i in range(len(cycle)):
                            u = cycle[i]
                            v = cycle[(i + 1) % len(cycle)]
                            edge = tuple(sorted((u, v)))
                            remaining_edges.discard(edge)
                            adj[u].discard(v)
                            adj[v].discard(u)
                        pieces.append(cycle)
                    else:
                        edge = next(iter(remaining_edges))
                        remaining_edges.discard(edge)
                        u, v = edge
                        adj[u].discard(v)
                        adj[v].discard(u)
                        pieces.append(list(edge))
            else:
                # No cycle found, take a single edge
                edge = next(iter(remaining_edges))
                remaining_edges.discard(edge)
                u, v = edge
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append(list(edge))
        
        return pieces
    
    def _find_any_cycle(adj, remaining_edges):
        """Find any cycle in the graph, or return None."""
        for start in list(adj.keys()):
            if not adj[start]:
                continue
            visited = set()
            parent = {}
            path = []
            
            def dfs(u):
                visited.add(u)
                path.append(u)
                for v in sorted(adj[u]):
                    if v == parent.get(u):
                        continue
                    if v in visited:
                        if v in path:
                            idx = path.index(v)
                            cycle = path[idx:]
                            if len(cycle) >= 3:
                                return cycle
                    else:
                        parent[v] = u
                        result = dfs(v)
                        if result:
                            return result
                path.pop()
                return None
            
            result = dfs(start)
            if result:
                return result
        return None

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
