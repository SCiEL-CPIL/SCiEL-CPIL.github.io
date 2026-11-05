import random
from collections import deque
from random import Random

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edge_list):
        if not edge_list:
            return []
        
        edge_set = {tuple(sorted(e)) for e in edge_list}
        n = max(max(e) for e in edge_set) + 1
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # 1. Find odd degree vertices
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 != 0]
        
        # 2. Greedily remove shortest paths between odd vertices to make graph Eulerian
        removed_edges = set()
        
        def get_shortest_path(u, v):
            if u == v:
                return []
            visited = {u}
            queue = deque([(u, [u])])
            while queue:
                node, path = queue.popleft()
                for nb in adj[node]:
                    if nb not in visited:
                        if nb == v:
                            return path + [v]
                        visited.add(nb)
                        queue.append((nb, path + [nb]))
            return None
            
        while len(odd_vertices) >= 2:
            best_pair = None
            best_dist = float('inf')
            best_path = None
            
            for i in range(len(odd_vertices)):
                for j in range(i+1, len(odd_vertices)):
                    u = odd_vertices[i]
                    v = odd_vertices[j]
                    path = get_shortest_path(u, v)
                    if path:
                        if len(path) - 1 < best_dist:
                            best_dist = len(path) - 1
                            best_pair = (u, v)
                            best_path = path
                            
            if best_pair is None:
                break
                
            u, v = best_pair
            for i in range(len(best_path) - 1):
                a, b = best_path[i], best_path[i+1]
                removed_edges.add((min(a,b), max(a,b)))
                
            odd_vertices.remove(u)
            odd_vertices.remove(v)
            
        # 3. Build remaining graph
        adj_rem = {i: set() for i in range(n)}
        remaining_edges = set()
        for e in edge_set:
            if e not in removed_edges:
                remaining_edges.add(e)
                u, v = e
                adj_rem[u].add(v)
                adj_rem[v].add(u)
                
        # 4. Decompose remaining graph into cycles
        cycles = []
        
        def find_longest_cycle(start):
            best = None
            visited = {start}
            path = [start]
            
            def dfs(node, path, visited):
                nonlocal best
                for nb in list(adj_rem[node]):
                    if nb == start and len(path) >= 3:
                        if best is None or len(path) > len(best):
                            best = path[:]
                    elif nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, path, visited)
                        path.pop()
                        visited.discard(nb)
                        
            dfs(start, path, visited)
            return best
            
        while remaining_edges:
            # Pick a start vertex
            # Heuristic: Start from a vertex with high degree in remaining graph
            start_candidates = [v for v in range(n) if len(adj_rem[v]) >= 2]
            if not start_candidates:
                # No cycles left? But remaining_edges is not empty.
                # This implies the graph is not Eulerian or we made a mistake.
                # Fallback: add as edges
                for e in remaining_edges:
                    cycles.append(list(e))
                break
                
            # Sort by degree to find long cycles
            start_candidates.sort(key=lambda x: len(adj_rem[x]), reverse=True)
            
            best_cycle = None
            # Try top 5 candidates
            for s in start_candidates[:5]:
                c = find_longest_cycle(s)
                if c is not None:
                    if best_cycle is None or len(c) > len(best_cycle):
                        best_cycle = c
                        
            if best_cycle is None:
                # Fallback
                for e in remaining_edges:
                    cycles.append(list(e))
                break
                
            # Remove best_cycle edges
            for i in range(len(best_cycle)):
                a = best_cycle[i]
                b = best_cycle[(i+1) % len(best_cycle)]
                key = (min(a,b), max(a,b))
                remaining_edges.discard(key)
                adj_rem[a].discard(b)
                adj_rem[b].discard(a)
                
            cycles.append(best_cycle)
            
        result = cycles
        for e in removed_edges:
            result.append(list(e))
            
        return result

    return [decompose(instance(*spec)) for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))]
