"""Path decomposition for Gallai's conjecture - combined approach with Warnsdorff heuristic."""
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
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bfs_path(adj, s, t):
        """Find shortest path from s to t in adj."""
        if s == t:
            return [s]
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        while queue:
            node = queue.popleft()
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    if neighbor == t:
                        path = []
                        cur = t
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        path.reverse()
                        return path
                    queue.append(neighbor)
        return None

    def find_eulerian_cycle(adj, start):
        """Find an Eulerian cycle starting/ending at start using Hierholzer's."""
        adj_copy = defaultdict(set)
        for u, vs in adj.items():
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
        return cycle

    def warnsdorff_path(adj, start, max_steps=None):
        """
        Find a long simple path using Warnsdorff's heuristic (greedy: always go to
        the neighbor with the fewest onward choices, preferring odd-degree vertices).
        """
        if max_steps is None:
            max_steps = sum(len(vs) for vs in adj.values())
        
        # Work with a copy of adjacency
        rem_adj = defaultdict(set)
        for u, vs in adj.items():
            rem_adj[u] = set(vs)
        
        path = [start]
        visited = {start}
        # Remove start from neighbors' adjacency
        for v in list(rem_adj.get(start, set())):
            rem_adj[v].discard(start)
        rem_adj[start] = set()
        
        current = start
        steps = 0
        while steps < max_steps:
            # Get available neighbors
            candidates = list(rem_adj.get(current, set()))
            if not candidates:
                break
            
            # Warnsdorff's rule: choose neighbor with fewest onward moves
            # Tie-break: prefer odd degree in original graph, then lower id
            def score(neighbor):
                onward = len(rem_adj.get(neighbor, set()))
                # Penalize even-degree vertices slightly to prefer odd ones
                deg = len(adj.get(neighbor, set()))
                parity_pen = 0 if deg % 2 == 1 else 0.1
                return (onward + parity_pen, neighbor)
            
            candidates.sort(key=score)
            next_v = candidates[0]
            
            path.append(next_v)
            visited.add(next_v)
            rem_adj[current].discard(next_v)
            rem_adj[next_v].discard(current)
            # Remove next_v from all its neighbors
            for v in list(rem_adj.get(next_v, set())):
                rem_adj[v].discard(next_v)
            rem_adj[next_v] = set()
            
            current = next_v
            steps += 1
        
        return path

    def decompose_graph(edges):
        """
        Decompose a graph into paths:
        1. Find all odd-degree vertices
        2. Pair them up, find paths between each pair using Warnsdorff
        3. Remove those edges
        4. Remaining graph has all even degrees -> decompose into Eulerian circuits
        5. Each circuit is a closed walk; convert to a path by removing the repeated endpoint
        """
        if not edges:
            return []
        
        adj = build_adj(edges)
        deg = {u: len(vs) for u, vs in adj.items()}
        odd_vertices = sorted([u for u, d in deg.items() if d % 2 == 1])
        
        remaining = set(edges)
        paths = []
        
        # Step 1: Pair odd vertices and find paths between them
        # Try to pair vertices that are close together to minimize path lengths
        for i in range(0, len(odd_vertices), 2):
            if i + 1 >= len(odd_vertices):
                break
            s, t = odd_vertices[i], odd_vertices[i + 1]
            
            # Build adjacency from remaining edges
            rem_adj = build_adj(remaining)
            
            # Try BFS first for shortest path
            path = bfs_path(rem_adj, s, t)
            
            if path is None:
                # Fallback: use Warnsdorff from s, hope we reach t
                path = warnsdorff_path(rem_adj, s)
                if path and path[-1] != t:
                    # Didn't reach t, try from t
                    path2 = warnsdorff_path(rem_adj, t)
                    if path2 and path2[-1] == s:
                        path = path2[::-1]
                    else:
                        # Just take whatever path we found
                        path = path
            
            if path is None or len(path) < 2:
                continue
            
            # Remove edges along this path
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
            
            if len(path) >= 2:
                paths.append(path)
        
        # Step 2: Decompose remaining graph (all even degrees) into Eulerian circuits
        while remaining:
            rem_adj = build_adj(remaining)
            
            # Find a vertex with non-zero degree
            start = None
            for u, vs in rem_adj.items():
                if vs:
                    start = u
                    break
            if start is None:
                break
            
            # Find Eulerian cycle
            cycle = find_eulerian_cycle(rem_adj, start)
            
            # Remove the cycle's edges from remaining
            for k in range(len(cycle) - 1):
                u, v = cycle[k], cycle[k + 1]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
            # The cycle is closed: last vertex connects back to first
            if len(cycle) > 1:
                u, v = cycle[-1], cycle[0]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
            
            # Convert cycle to path: remove the last vertex (which equals the first)
            if len(cycle) > 1:
                path = cycle[:-1] if cycle[0] == cycle[-1] else cycle
                if len(path) >= 2:
                    paths.append(path)
        
        return paths

    def decompose_greedy(edges):
        """
        Greedy approach: repeatedly extract the longest simple path using Warnsdorff.
        This tends to produce fewer, longer paths.
        """
        remaining = set(edges)
        paths = []
        
        while remaining:
            rem_adj = build_adj(remaining)
            
            # Find a good starting vertex: prefer odd-degree, then highest degree
            start_candidates = []
            for u, vs in rem_adj.items():
                if vs:
                    deg = len(vs)
                    # Score: prefer odd degree, then higher degree
                    score = (0 if deg % 2 == 1 else 1, -deg, u)
                    start_candidates.append((score, u))
            
            if not start_candidates:
                break
            
            start_candidates.sort(key=lambda x: x[0])
            start = start_candidates[0][1]
            
            # Use Warnsdorff to find a long path
            path = warnsdorff_path(rem_adj, start)
            
            if path is None or len(path) < 2:
                # Fallback: single edge
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.discard(e)
                continue
            
            # Remove edges along this path
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
            
            paths.append(path)
        
        return paths

    def verify_decomposition(edges, paths):
        """Verify that paths form a valid edge-disjoint decomposition covering all edges."""
        used = set()
        for path in paths:
            # Check simplicity
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = (min(u, v), max(u, v))
                if e in used:
                    return False
                if e not in edges:
                    return False
                used.add(e)
        return used == set(edges)

    def local_search_refine(edges, paths, time_limit=5.0):
        """
        Try to reduce the number of paths by merging adjacent paths.
        Two paths can be merged if the end of one connects to the start of the other
        (or vice versa) via an unused edge, and the merged walk is simple.
        """
        deadline = time.time() + time_limit
        current_paths = [list(p) for p in paths]
        
        while time.time() < deadline:
            improved = False
            n = len(current_paths)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(n):
                    if i == j:
                        continue
                    # Try to merge path j into path i
                    # Check if end of i connects to start of j
                    if len(current_paths[i]) < 2 or len(current_paths[j]) < 2:
                        continue
                    
                    # Try: end of i -> start of j
                    a, b = current_paths[i][-1], current_paths[j][0]
                    e = (min(a, b), max(a, b))
                    if e in edges:
                        # Check if e is not already used in either path
                        used_edges_i = set()
                        for k in range(len(current_paths[i]) - 1):
                            u, v = current_paths[i][k], current_paths[i][k+1]
                            used_edges_i.add((min(u,v), max(u,v)))
                        used_edges_j = set()
                        for k in range(len(current_paths[j]) - 1):
                            u, v = current_paths[j][k], current_paths[j][k+1]
                            used_edges_j.add((min(u,v), max(u,v)))
                        
                        if e not in used_edges_i and e not in used_edges_j:
                            merged = current_paths[i] + current_paths[j][1:]
                            # Check simplicity
                            if len(set(merged)) == len(merged):
                                # Check all edges are valid
                                valid = True
                                for k in range(len(merged) - 1):
                                    u, v = merged[k], merged[k+1]
                                    ee = (min(u,v), max(u,v))
                                    if ee not in edges:
                                        valid = False
                                        break
                                if valid:
                                    new_paths = [p for idx, p in enumerate(current_paths) if idx != i and idx != j]
                                    new_paths.append(merged)
                                    current_paths = new_paths
                                    improved = True
                                    break
                    if improved:
                        break
                    
                    # Try: end of j -> start of i
                    a, b = current_paths[j][-1], current_paths[i][0]
                    e = (min(a, b), max(a, b))
                    if e in edges:
                        used_edges_i = set()
                        for k in range(len(current_paths[i]) - 1):
                            u, v = current_paths[i][k], current_paths[i][k+1]
                            used_edges_i.add((min(u,v), max(u,v)))
                        used_edges_j = set()
                        for k in range(len(current_paths[j]) - 1):
                            u, v = current_paths[j][k], current_paths[j][k+1]
                            used_edges_j.add((min(u,v), max(u,v)))
                        
                        if e not in used_edges_i and e not in used_edges_j:
                            merged = current_paths[j] + current_paths[i][1:]
                            if len(set(merged)) == len(merged):
                                valid = True
                                for k in range(len(merged) - 1):
                                    u, v = merged[k], merged[k+1]
                                    ee = (min(u,v), max(u,v))
                                    if ee not in edges:
                                        valid = False
                                        break
                                if valid:
                                    new_paths = [p for idx, p in enumerate(current_paths) if idx != i and idx != j]
                                    new_paths.append(merged)
                                    current_paths = new_paths
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current_paths

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
        edge_set = set(edges)
        
        # Try multiple decomposition strategies and pick the best
        candidates = []
        
        # Strategy 1: Odd-pairing + Eulerian circuits
        try:
            if time.time() < DEADLINE - 20:
                paths1 = decompose_graph(edges)
                if verify_decomposition(edge_set, paths1):
                    candidates.append(paths1)
        except Exception:
            pass
        
        # Strategy 2: Greedy Warnsdorff
        try:
            if time.time() < DEADLINE - 20:
                paths2 = decompose_greedy(edges)
                if verify_decomposition(edge_set, paths2):
                    candidates.append(paths2)
        except Exception:
            pass
        
        # Pick the best candidate (fewest paths)
        if candidates:
            candidates.sort(key=len)
            best_paths = candidates[0]
            
            # Try local search refinement
            if time.time() < DEADLINE - 10:
                refined = local_search_refine(edge_set, best_paths, time_limit=5.0)
                if verify_decomposition(edge_set, refined):
                    best_paths = refined
            
            results.append(best_paths)
        else:
            # Fallback to single edges
            results.append([[u, v] for u, v in edges])
    
    return results
