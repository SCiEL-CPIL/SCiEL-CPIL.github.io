"""Path decomposition for Gallai's conjecture - improved multi-strategy approach."""
import time
from collections import defaultdict, deque
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

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

    def bfs_all_pairs(adj, sources):
        """BFS from multiple sources to find shortest paths."""
        results = {}
        for s in sources:
            if s == s:
                results[(s, s)] = [s]
                continue
            visited = {s}
            parent = {s: None}
            queue = deque([s])
            while queue:
                node = queue.popleft()
                for neighbor in adj.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = node
                        queue.append(neighbor)
            for t in sources:
                if t in parent:
                    path = []
                    cur = t
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    results[(s, t)] = path
        return results

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

    def warnsdorff_path(adj, start, max_steps=None, rng=None):
        """
        Find a long simple path using Warnsdorff's heuristic.
        """
        if max_steps is None:
            max_steps = sum(len(vs) for vs in adj.values())
        if rng is None:
            rng = random.Random()
        
        rem_adj = defaultdict(set)
        for u, vs in adj.items():
            rem_adj[u] = set(vs)
        
        path = [start]
        visited = {start}
        for v in list(rem_adj.get(start, set())):
            rem_adj[v].discard(start)
        rem_adj[start] = set()
        
        current = start
        steps = 0
        while steps < max_steps:
            candidates = list(rem_adj.get(current, set()))
            if not candidates:
                break
            
            def score(neighbor):
                onward = len(rem_adj.get(neighbor, set()))
                deg = len(adj.get(neighbor, set()))
                parity_pen = 0 if deg % 2 == 1 else 0.1
                return (onward + parity_pen, neighbor)
            
            candidates.sort(key=score)
            next_v = candidates[0]
            
            path.append(next_v)
            visited.add(next_v)
            rem_adj[current].discard(next_v)
            rem_adj[next_v].discard(current)
            for v in list(rem_adj.get(next_v, set())):
                rem_adj[v].discard(next_v)
            rem_adj[next_v] = set()
            
            current = next_v
            steps += 1
        
        return path

    def decompose_with_odd_pairing(edges, rng):
        """
        Decompose a graph into paths:
        1. Find all odd-degree vertices
        2. Pair them up using shortest paths
        3. Remove those edges
        4. Remaining graph has all even degrees -> decompose into Eulerian circuits
        5. Each circuit is a closed walk; convert to a path
        """
        if not edges:
            return []
        
        adj = build_adj(edges)
        deg = {u: len(vs) for u, vs in adj.items()}
        odd_vertices = sorted([u for u, d in deg.items() if d % 2 == 1])
        
        remaining = set(edges)
        paths = []
        
        # Step 1: Pair odd vertices and find shortest paths between them
        for i in range(0, len(odd_vertices), 2):
            if i + 1 >= len(odd_vertices):
                break
            s, t = odd_vertices[i], odd_vertices[i + 1]
            
            rem_adj = build_adj(remaining)
            
            # BFS for shortest path
            visited = {s}
            parent = {s: None}
            queue = deque([s])
            path = None
            while queue:
                node = queue.popleft()
                for neighbor in rem_adj.get(node, []):
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
                            break
                        queue.append(neighbor)
                if path is not None:
                    break
            
            if path is None or len(path) < 2:
                continue
            
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
            
            paths.append(path)
        
        # Step 2: Decompose remaining graph into Eulerian circuits
        while remaining:
            rem_adj = build_adj(remaining)
            
            start = None
            for u, vs in rem_adj.items():
                if vs:
                    start = u
                    break
            if start is None:
                break
            
            cycle = find_eulerian_cycle(rem_adj, start)
            
            for k in range(len(cycle) - 1):
                u, v = cycle[k], cycle[k + 1]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
            if len(cycle) > 1:
                u, v = cycle[-1], cycle[0]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
            
            if len(cycle) > 1:
                path = cycle[:-1] if cycle[0] == cycle[-1] else cycle
                if len(path) >= 2:
                    paths.append(path)
        
        return paths

    def decompose_greedy_warnsdorff(edges, rng, max_restarts=5):
        """
        Greedy approach with multiple restarts: repeatedly extract the longest
        simple path using Warnsdorff from different starting vertices.
        """
        remaining = set(edges)
        paths = []
        
        while remaining:
            rem_adj = build_adj(remaining)
            
            # Collect candidate starting vertices
            start_candidates = []
            for u, vs in rem_adj.items():
                if vs:
                    deg = len(vs)
                    start_candidates.append((u, deg))
            
            if not start_candidates:
                break
            
            # Sort by degree (prefer high degree, odd degree)
            start_candidates.sort(key=lambda x: (-x[1], x[0]))
            
            best_path = None
            best_len = 0
            
            # Try multiple starting vertices
            for u, deg in start_candidates[:max_restarts]:
                path = warnsdorff_path(rem_adj, u, rng=rng)
                if path and len(path) > best_len:
                    best_path = path
                    best_len = len(path)
            
            if best_path is None or len(best_path) < 2:
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.discard(e)
                continue
            
            for k in range(len(best_path) - 1):
                u, v = best_path[k], best_path[k + 1]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
            
            paths.append(best_path)
        
        return paths

    def decompose_hybrid(edges, rng):
        """
        Hybrid: use odd-pairing for odd vertices, then greedy Warnsdorff for the rest.
        """
        adj = build_adj(edges)
        deg = {u: len(vs) for u, vs in adj.items()}
        odd_vertices = sorted([u for u, d in deg.items() if d % 2 == 1])
        
        remaining = set(edges)
        paths = []
        
        # Pair odd vertices with shortest paths
        for i in range(0, len(odd_vertices), 2):
            if i + 1 >= len(odd_vertices):
                break
            s, t = odd_vertices[i], odd_vertices[i + 1]
            
            rem_adj = build_adj(remaining)
            visited = {s}
            parent = {s: None}
            queue = deque([s])
            path = None
            while queue:
                node = queue.popleft()
                for neighbor in rem_adj.get(node, []):
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
                            break
                        queue.append(neighbor)
                if path is not None:
                    break
            
            if path is None or len(path) < 2:
                continue
            
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
            
            paths.append(path)
        
        # Now decompose remaining with greedy Warnsdorff
        while remaining:
            rem_adj = build_adj(remaining)
            start_candidates = []
            for u, vs in rem_adj.items():
                if vs:
                    start_candidates.append((u, len(vs)))
            
            if not start_candidates:
                break
            
            start_candidates.sort(key=lambda x: (-x[1], x[0]))
            
            best_path = None
            best_len = 0
            for u, d in start_candidates[:3]:
                path = warnsdorff_path(rem_adj, u, rng=rng)
                if path and len(path) > best_len:
                    best_path = path
                    best_len = len(path)
            
            if best_path is None or len(best_path) < 2:
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.discard(e)
                continue
            
            for k in range(len(best_path) - 1):
                u, v = best_path[k], best_path[k + 1]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
            
            paths.append(best_path)
        
        return paths

    def verify_decomposition(edges, paths):
        """Verify that paths form a valid edge-disjoint decomposition covering all edges."""
        used = set()
        for path in paths:
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
        Try to reduce the number of paths by:
        1. Merging paths that can be connected
        2. Splitting long paths to enable more merges
        """
        deadline = time.time() + time_limit
        current_paths = [list(p) for p in paths]
        
        def path_edges(path):
            es = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                es.add((min(u, v), max(u, v)))
            return es
        
        def try_merge(paths_list, i, j, direction):
            """Try to merge path j into path i (or vice versa)."""
            p_i = paths_list[i]
            p_j = paths_list[j]
            
            if len(p_i) < 2 or len(p_j) < 2:
                return None
            
            if direction == 0:
                # end of i -> start of j
                a, b = p_i[-1], p_j[0]
            else:
                # end of j -> start of i
                a, b = p_j[-1], p_i[0]
            
            e = (min(a, b), max(a, b))
            if e not in edges:
                return None
            
            used_i = path_edges(p_i)
            used_j = path_edges(p_j)
            if e in used_i or e in used_j:
                return None
            
            if direction == 0:
                merged = p_i + p_j[1:]
            else:
                merged = p_j + p_i[1:]
            
            if len(set(merged)) != len(merged):
                return None
            
            valid = True
            for k in range(len(merged) - 1):
                u, v = merged[k], merged[k + 1]
                ee = (min(u, v), max(u, v))
                if ee not in edges:
                    valid = False
                    break
            if not valid:
                return None
            
            return merged
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(current_paths)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(n):
                    if i == j:
                        continue
                    
                    for direction in [0, 1]:
                        merged = try_merge(current_paths, i, j, direction)
                        if merged is not None:
                            new_paths = [p for idx, p in enumerate(current_paths) if idx != i and idx != j]
                            new_paths.append(merged)
                            current_paths = new_paths
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return current_paths

    def split_and_merge(edges, paths, time_limit=3.0):
        """
        Try splitting long paths at various points to enable more merges.
        """
        deadline = time.time() + time_limit
        current_paths = [list(p) for p in paths]
        
        best = current_paths
        best_count = len(best)
        
        # Try splitting each path at each possible point
        for idx in range(len(current_paths)):
            if time.time() > deadline:
                break
            p = current_paths[idx]
            if len(p) < 4:
                continue
            
            for split_point in range(1, len(p) - 1):
                part1 = p[:split_point + 1]
                part2 = p[split_point:]
                
                if len(part1) < 2 or len(part2) < 2:
                    continue
                
                # Check if split creates valid paths
                if len(set(part1)) != len(part1) or len(set(part2)) != len(part2):
                    continue
                
                new_paths = [p for i, p in enumerate(current_paths) if i != idx]
                new_paths.append(part1)
                new_paths.append(part2)
                
                # Try to merge
                refined = local_search_refine(edges, new_paths, time_limit=1.0)
                if verify_decomposition(edges, refined) and len(refined) < best_count:
                    best = refined
                    best_count = len(refined)
                    current_paths = refined
                    break
        
        return best

    # Main logic
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > DEADLINE - 20:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(n, p, seed)
        edge_set = set(edges)
        
        candidates = []
        
        # Strategy 1: Odd-pairing + Eulerian circuits
        try:
            if time.time() < DEADLINE - 25:
                rng1 = random.Random(seed * 1000 + 1)
                paths1 = decompose_with_odd_pairing(edges, rng1)
                if verify_decomposition(edge_set, paths1):
                    candidates.append(paths1)
        except Exception:
            pass
        
        # Strategy 2: Greedy Warnsdorff with restarts
        try:
            if time.time() < DEADLINE - 25:
                rng2 = random.Random(seed * 1000 + 2)
                paths2 = decompose_greedy_warnsdorff(edges, rng2, max_restarts=5)
                if verify_decomposition(edge_set, paths2):
                    candidates.append(paths2)
        except Exception:
            pass
        
        # Strategy 3: Hybrid
        try:
            if time.time() < DEADLINE - 25:
                rng3 = random.Random(seed * 1000 + 3)
                paths3 = decompose_hybrid(edges, rng3)
                if verify_decomposition(edge_set, paths3):
                    candidates.append(paths3)
        except Exception:
            pass
        
        # Strategy 4: Greedy with different restart count
        try:
            if time.time() < DEADLINE - 25:
                rng4 = random.Random(seed * 1000 + 4)
                paths4 = decompose_greedy_warnsdorff(edges, rng4, max_restarts=10)
                if verify_decomposition(edge_set, paths4):
                    candidates.append(paths4)
        except Exception:
            pass
        
        # Pick the best candidate
        if candidates:
            candidates.sort(key=len)
            best_paths = candidates[0]
            
            # Local search refinement
            if time.time() < DEADLINE - 15:
                refined = local_search_refine(edge_set, best_paths, time_limit=4.0)
                if verify_decomposition(edge_set, refined):
                    best_paths = refined
            
            # Try split and merge
            if time.time() < DEADLINE - 10:
                split_merged = split_and_merge(edge_set, best_paths, time_limit=3.0)
                if verify_decomposition(edge_set, split_merged) and len(split_merged) < len(best_paths):
                    best_paths = split_merged
            
            results.append(best_paths)
        else:
            results.append([[u, v] for u, v in edges])
    
    return results
