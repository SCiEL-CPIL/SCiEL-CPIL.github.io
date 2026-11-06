"""Rewritten program for Gallai's conjecture path decomposition."""
import time
from collections import defaultdict, deque
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)
    
    def build_graph(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def degree_map(adj):
        return {u: len(v) for u, v in adj.items()}
    
    def warnsdorff_path(rem_adj, start, max_len=None):
        """Find a long path using Warnsdorff's rule: prefer neighbors with fewer remaining edges."""
        path = [start]
        visited = {start}
        while True:
            current = path[-1]
            neighbors = [n for n in rem_adj.get(current, set()) if n not in visited]
            if not neighbors:
                break
            if max_len and len(path) >= max_len:
                break
            # Warnsdorff: pick neighbor with fewest unvisited neighbors
            neighbors.sort(key=lambda n: len([x for x in rem_adj.get(n, set()) if x not in visited]))
            next_node = neighbors[0]
            path.append(next_node)
            visited.add(next_node)
        return path
    
    def greedy_maximal_path_decomposition(edges, adj, num_restarts=5):
        """
        Greedy approach with Warnsdorff's rule and multiple restarts.
        Repeatedly find long paths using heuristic, keeping the best result.
        """
        best_paths = None
        best_count = float('inf')
        
        for restart in range(num_restarts):
            remaining_edges = set(edges)
            paths = []
            
            rem_adj = defaultdict(set)
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Determine starting vertices: prioritize odd-degree vertices
            odd_vertices = [u for u, vs in rem_adj.items() if len(vs) % 2 == 1]
            
            while remaining_edges:
                # Choose start vertex
                if odd_vertices:
                    # Pick an odd vertex with highest degree first
                    odd_vertices.sort(key=lambda u: len(rem_adj.get(u, set())), reverse=True)
                    start = odd_vertices[0]
                    odd_vertices = [u for u in odd_vertices if u != start]
                else:
                    # Pick vertex with highest degree
                    all_verts = [u for u, vs in rem_adj.items() if vs]
                    if not all_verts:
                        break
                    all_verts.sort(key=lambda u: len(rem_adj.get(u, set())), reverse=True)
                    start = all_verts[0]
                
                # Find path using Warnsdorff
                path = warnsdorff_path(rem_adj, start)
                
                if len(path) < 2:
                    # Fallback: take single edge
                    e = next(iter(remaining_edges))
                    paths.append([e[0], e[1]])
                    remaining_edges.remove(e)
                    u, v = e
                    if v in rem_adj[u]:
                        rem_adj[u].discard(v)
                    if u in rem_adj[v]:
                        rem_adj[v].discard(u)
                    if not rem_adj[u]:
                        del rem_adj[u]
                    if not rem_adj[v]:
                        del rem_adj[v]
                    continue
                
                # Remove path edges
                path_edges = set()
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        path_edges.add(e)
                
                if not path_edges:
                    e = next(iter(remaining_edges))
                    paths.append([e[0], e[1]])
                    remaining_edges.remove(e)
                    u, v = e
                    if v in rem_adj[u]:
                        rem_adj[u].discard(v)
                    if u in rem_adj[v]:
                        rem_adj[v].discard(u)
                    if not rem_adj[u]:
                        del rem_adj[u]
                    if not rem_adj[v]:
                        del rem_adj[v]
                    continue
                
                paths.append(path)
                remaining_edges -= path_edges
                for u, v in path_edges:
                    if v in rem_adj[u]:
                        rem_adj[u].discard(v)
                    if u in rem_adj[v]:
                        rem_adj[v].discard(u)
                    if not rem_adj[u]:
                        del rem_adj[u]
                    if not rem_adj[v]:
                        del rem_adj[v]
                
                # Update odd vertices
                odd_vertices = [u for u, vs in rem_adj.items() if len(vs) % 2 == 1]
            
            # Check validity
            if remaining_edges:
                continue
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
            
            # Early stop if we hit theoretical minimum
            n_odd = sum(1 for u, d in degree_map(adj).items() if d % 2 == 1)
            min_possible = max(n_odd // 2, 1)
            if best_count <= min_possible:
                break
        
        if best_paths is None:
            # Fallback: single edges
            return [[u, v] for u, v in edges]
        
        return best_paths
    
    def improved_decomposition(edges, adj):
        """
        Hybrid approach:
        1. Pair odd vertices with greedy long paths
        2. Decompose remaining even-degree graph into Eulerian cycles
        """
        deg = degree_map(adj)
        odd_vertices = [u for u, d in deg.items() if d % 2 == 1]
        
        remaining_edges = set(edges)
        paths = []
        
        # Pair odd vertices using greedy long paths
        # Sort by degree descending for better pairing
        odd_vertices.sort(key=lambda u: deg.get(u, 0), reverse=True)
        
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        i = 0
        while i < len(odd_vertices):
            if i + 1 >= len(odd_vertices):
                break
            s, t = odd_vertices[i], odd_vertices[i + 1]
            
            # Find a path from s to t using Warnsdorff-like heuristic
            # Try to find a long path from s that eventually reaches t
            path = [s]
            visited = {s}
            current = s
            success = False
            
            while True:
                neighbors = [n for n in rem_adj.get(current, set()) if n not in visited]
                if not neighbors:
                    break
                if current == t:
                    success = True
                    break
                # Prefer moving toward t if close, otherwise Warnsdorff
                # Simple heuristic: pick neighbor with fewest options
                neighbors.sort(key=lambda n: len([x for x in rem_adj.get(n, set()) if x not in visited]))
                # But if t is a neighbor, go there
                if t in neighbors:
                    next_node = t
                else:
                    next_node = neighbors[0]
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if success and len(path) >= 2:
                # Remove path edges
                for k in range(len(path) - 1):
                    u, v = path[k], path[k + 1]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        if v in rem_adj[u]:
                            rem_adj[u].discard(v)
                        if u in rem_adj[v]:
                            rem_adj[v].discard(u)
                        if not rem_adj[u]:
                            del rem_adj[u]
                        if not rem_adj[v]:
                            del rem_adj[v]
                paths.append(path)
            
            i += 2
        
        # Now decompose remaining into Eulerian cycles
        while remaining_edges:
            rem_adj = defaultdict(set)
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            start = next((u for u, vs in rem_adj.items() if vs), None)
            if start is None:
                break
            
            adj_copy = defaultdict(set)
            for u, vs in rem_adj.items():
                adj_copy[u] = set(vs)
            
            stack = [start]
            cycle = []
            while stack:
                v = stack[-1]
                if adj_copy[v]:
                    w = next(iter(adj_copy[v]))
                    adj_copy[v].discard(w)
                    adj_copy[w].discard(v)
                    stack.append(w)
                else:
                    cycle.append(stack.pop())
            
            cycle.reverse()
            
            if len(cycle) > 1:
                # Remove cycle edges
                for k in range(len(cycle) - 1):
                    u, v = cycle[k], cycle[k + 1]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                u, v = cycle[-1], cycle[0]
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    remaining_edges.remove(e)
                
                paths.append(cycle)
        
        return paths
    
    def validate_paths(paths, edges, adj):
        """Validate that paths are edge-disjoint, use graph edges, and cover all edges."""
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = (min(u, v), max(u, v))
                if e not in adj.get(u, set()) or e in used_edges:
                    return False
                used_edges.add(e)
        return used_edges == set(edges)
    
    # Main logic
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > DEADLINE - 15:
            # Fallback: single edges
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(n, p, seed)
        adj = build_graph(edges)
        
        # Try multiple strategies and keep the best
        candidates = []
        
        # Strategy 1: Improved decomposition (pair odd + Eulerian cycles)
        try:
            paths1 = improved_decomposition(edges, adj)
            if validate_paths(paths1, edges, adj):
                candidates.append(paths1)
        except Exception:
            pass
        
        # Strategy 2: Greedy Warnsdorff with restarts
        try:
            paths2 = greedy_maximal_path_decomposition(edges, adj, num_restarts=3)
            if validate_paths(paths2, edges, adj):
                candidates.append(paths2)
        except Exception:
            pass
        
        # Strategy 3: Single edges (always valid fallback)
        paths3 = [[u, v] for u, v in edges]
        candidates.append(paths3)
        
        # Pick the best (fewest paths)
        best = min(candidates, key=len)
        results.append(best)
    
    return results
