import time
from random import Random
import sys
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # 5s safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_components(edges, n):
        """Returns a list of sets of edges for each connected component."""
        if not edges:
            return []
        
        # Build adjacency for component finding
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        visited = set()
        components = []
        
        for start in adj:
            if start in visited:
                continue
            # BFS to find component
            q = deque([start])
            comp_vertices = set()
            comp_edges = set()
            visited.add(start)
            while q:
                u = q.popleft()
                comp_vertices.add(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
                    # Add edge if both in component
                    e = tuple(sorted((u, v)))
                    comp_edges.add(e)
            
            components.append(comp_edges)
        
        return components

    def find_paths_between(u, v, edges, n, max_paths=5):
        """
        Finds up to max_paths simple paths between u and v using edges.
        Returns a list of lists of vertices.
        """
        if u == v:
            return [] # Simple path between distinct vertices
        
        paths = []
        # Sort neighbors to ensure determinism and potentially better heuristics
        adj = defaultdict(set)
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
            
        if v not in adj[u]:
            return []
            
        def dfs(current, target, visited, path):
            if len(paths) >= max_paths:
                return
            
            if current == target:
                paths.append(path[:])
                return
                
            # Heuristic: visit neighbors with lower degree in remaining graph first?
            # For now, just standard DFS
            for nb in sorted(adj[current]):
                if nb not in visited:
                    # Check if edge (current, nb) is in edges
                    e = tuple(sorted((current, nb)))
                    if e in edges:
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, target, visited, path)
                        path.pop()
                        visited.remove(nb)
                        
                        if len(paths) >= max_paths:
                            return

        dfs(u, v, {u}, [u])
        return paths

    def solve_component(edges, n, time_limit):
        """
        Solves a single connected component.
        Returns a list of paths or None if failed.
        """
        if not edges:
            return []
        
        # Check time
        if time.time() > time_limit:
            return None

        # Calculate odd vertices
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
            
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        
        # If no odd vertices, we can pick any pair of vertices to be endpoints
        # Heuristic: pick pairs that are far apart? Or just arbitrary.
        # Let's pick pairs from the edges themselves to ensure connectivity.
        if not odd_vertices:
            # Get all vertices involved
            verts = set()
            for u, v in edges:
                verts.add(u)
                verts.add(v)
            verts = list(verts)
            # Try a few random pairs
            candidates = []
            if len(verts) >= 2:
                # Pick first two
                candidates.append((verts[0], verts[1]))
                # Pick a few more
                for i in range(min(5, len(verts)-1)):
                    candidates.append((verts[i], verts[i+1]))
        else:
            # Must pick from odd vertices
            candidates = []
            for i in range(len(odd_vertices)):
                for j in range(i+1, len(odd_vertices)):
                    candidates.append((odd_vertices[i], odd_vertices[j]))
                    
        # Shuffle candidates to add randomness or use deterministic order?
        # Deterministic is better for reproducibility, but we can try all.
        
        for u, v in candidates:
            # Find paths between u and v
            # Note: find_paths_between expects edges as a set
            paths_uv = find_paths_between(u, v, set(edges), n, max_paths=10)
            
            for path in paths_uv:
                # Convert path to set of edges
                path_edges = set()
                for i in range(len(path)-1):
                    path_edges.add(tuple(sorted((path[i], path[i+1]))))
                    
                # Check if path is valid (all edges in edges)
                if not path_edges.issubset(edges):
                    continue
                    
                # Remove path edges
                remaining_edges = edges - path_edges
                
                # Determine new odd vertices for remaining graph
                # We can just recurse, which will re-calculate
                # But we need to ensure we don't exceed time
                
                sub_result = solve_component(remaining_edges, n, time_limit)
                if sub_result is not None:
                    return [path] + sub_result
                    
        return None

    def greedy_path_cover(edges, n):
        """
        Fallback greedy algorithm.
        """
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            # Find longest simple path
            # Build adj
            adj = defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
                
            # Find start: min degree
            min_deg = min(len(adj[v]) for v in adj if adj[v]) if adj else 0
            candidates = [v for v in adj if len(adj[v]) == min_deg]
            if not candidates:
                break
                
            best_path = []
            best_edges = set()
            
            for start in candidates:
                path = [start]
                visited = {start}
                current = start
                used_edges = set()
                
                while True:
                    next_candidates = [nb for nb in adj[current] if nb not in visited]
                    if not next_candidates:
                        break
                    
                    # Heuristic: min dynamic degree
                    best_next = None
                    min_dyn_deg = float('inf')
                    for nb in next_candidates:
                        dyn_deg = sum(1 for x in adj[nb] if x not in visited)
                        if dyn_deg < min_dyn_deg:
                            min_dyn_deg = dyn_deg
                            best_next = nb
                            
                    if best_next is None:
                        break
                        
                    u, v = sorted((current, best_next))
                    used_edges.add((u, v))
                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next
                    
                if len(path) > len(best_path):
                    best_path = path
                    best_edges = used_edges
                    
            if not best_path:
                # Fallback: single edge
                e = next(iter(remaining_edges))
                paths.append(list(e))
                remaining_edges.discard(e)
            else:
                paths.append(best_path)
                remaining_edges.difference_update(best_edges)
                
            if time.time() > DEADLINE:
                for e in remaining_edges:
                    paths.append(list(e))
                break
                
        return paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        if time.time() > DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        n = spec[0]
        
        # 1. Get Greedy Solution
        greedy_paths = greedy_path_cover(edges, n)
        greedy_count = len(greedy_paths)
        
        # 2. Calculate Lower Bound
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        odd_count = sum(1 for v in degrees if degrees[v] % 2 != 0)
        lb_odd = odd_count // 2 if odd_count > 0 else 1
        lb_len = (len(edges) + n - 2) // (n - 1) if n > 1 else 1
        lower_bound = max(lb_odd, lb_len)
        
        best_paths = greedy_paths
        
        # 3. Try to improve with Search if Greedy is not optimal
        if greedy_count > lower_bound:
            # Allocate time for search
            # Roughly 1/3 of remaining time per graph?
            # We have ~115s total.
            # Let's give 30s to search for each graph if needed.
            search_deadline = time.time() + 30
            
            # Solve components
            comps = get_components(edges, n)
            search_results = []
            success = True
            
            for comp_edges in comps:
                res = solve_component(comp_edges, n, search_deadline)
                if res is None:
                    success = False
                    break
                search_results.extend(res)
                
            if success and len(search_results) < greedy_count:
                best_paths = search_results
                
        results.append(best_paths)
        
    return results

