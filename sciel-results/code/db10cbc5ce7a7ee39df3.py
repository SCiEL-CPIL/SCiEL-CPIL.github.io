"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _greedy_walk(start, remaining, adj):
        """Greedily walk from start, preferring neighbors with fewer options (dead-end avoidance)."""
        path = [start]
        visited = {start}
        current = start
        while True:
            candidates = []
            for nb in adj[current]:
                if nb not in visited:
                    e = tuple(sorted((current, nb)))
                    if e in remaining:
                        # Count unvisited neighbors of nb
                        opts = sum(1 for x in adj[nb] if x not in visited)
                        candidates.append((opts, nb))
            if not candidates:
                break
            # Pick neighbor with fewest options (to avoid trapping)
            candidates.sort(key=lambda x: x[0])
            next_v = candidates[0][1]
            path.append(next_v)
            visited.add(next_v)
            current = next_v
        return path

    def _greedy_walk_wide(start, remaining, adj):
        """Greedily walk from start, preferring neighbors with more options (keep path open)."""
        path = [start]
        visited = {start}
        current = start
        while True:
            candidates = []
            for nb in adj[current]:
                if nb not in visited:
                    e = tuple(sorted((current, nb)))
                    if e in remaining:
                        opts = sum(1 for x in adj[nb] if x not in visited)
                        candidates.append((opts, nb))
            if not candidates:
                break
            # Pick neighbor with most options
            candidates.sort(key=lambda x: -x[0])
            next_v = candidates[0][1]
            path.append(next_v)
            visited.add(next_v)
            current = next_v
        return path

    def _find_long_path(start, remaining, adj, max_nodes=40):
        """Find a long simple path using beam search."""
        beam = [([start], {start})]
        best = [start]
        
        for _ in range(max_nodes):
            new_beam = []
            for path, visited in beam:
                current = path[-1]
                candidates = []
                for nb in adj[current]:
                    if nb not in visited:
                        e = tuple(sorted((current, nb)))
                        if e in remaining:
                            opts = sum(1 for x in adj[nb] if x not in visited)
                            candidates.append((opts, nb))
                if not candidates:
                    if len(path) > len(best):
                        best = list(path)
                    continue
                # Keep top 3 candidates per path
                candidates.sort(key=lambda x: x[0])
                for opts, nb in candidates[:3]:
                    new_path = path + [nb]
                    new_visited = visited | {nb}
                    new_beam.append((new_path, new_visited))
                    if len(new_path) > len(best):
                        best = list(new_path)
            if not new_beam:
                break
            # Limit beam size
            new_beam.sort(key=lambda x: -len(x[0]))
            beam = new_beam[:20]
        
        return best

    def _path_decomposition(edges, max_time=None):
        """Decompose edges into edge-disjoint paths minimizing count."""
        if max_time is None:
            max_time = deadline - 5
        
        remaining = set(edges)
        adj = _build_adj(edges)
        paths = []
        
        while remaining:
            if time.time() > max_time:
                break
            
            # Compute degrees in remaining graph
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            # Identify odd-degree vertices
            odd_verts = [v for v, d in deg.items() if d % 2 == 1]
            
            # Choose starting vertices: prefer odd-degree or low-degree
            candidates = []
            for v in deg:
                if v in odd_verts:
                    candidates.append((0, v))
                elif deg[v] <= 2:
                    candidates.append((1, v))
                else:
                    candidates.append((2, v))
            
            random.shuffle(candidates)
            candidates.sort(key=lambda x: x[0])
            
            best_path = None
            best_len = 0
            
            # Try multiple starts with different strategies
            for _, start in candidates[:20]:
                # Try narrow greedy
                p1 = _greedy_walk(start, remaining, adj)
                if len(p1) > best_len:
                    best_len = len(p1)
                    best_path = p1
                
                # Try wide greedy
                p2 = _greedy_walk_wide(start, remaining, adj)
                if len(p2) > best_len:
                    best_len = len(p2)
                    best_path = p2
                
                # Try beam search for promising starts
                if deg[start] <= 3 or start in odd_verts:
                    p3 = _find_long_path(start, remaining, adj, max_nodes=30)
                    if len(p3) > best_len:
                        best_len = len(p3)
                        best_path = p3
            
            if best_path is None:
                e = next(iter(remaining))
                best_path = list(e)
            
            paths.append(best_path)
            
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                remaining.discard(e)
        
        # Fallback for any remaining edges
        if remaining:
            for e in remaining:
                paths.append(list(e))
        
        return paths

    def _local_search_improve(paths, edges, max_time=None):
        """Try to merge paths or rearrange to reduce count."""
        if max_time is None:
            max_time = deadline - 3
        
        edge_set = set(edges)
        adj = _build_adj(edges)
        
        # Build edge-to-path mapping
        edge_to_path = {}
        for pi, path in enumerate(paths):
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                edge_to_path[e] = pi
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try to merge two paths if they share a vertex and can be reconnected
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    if time.time() > max_time:
                        break
                    pi = paths[i]
                    pj = paths[j]
                    # Check if we can merge: pi ends at v, pj starts at v (or vice versa)
                    # Try all pairs of endpoints
                    endpoints_i = {pi[0], pi[-1]}
                    endpoints_j = {pj[0], pj[-1]}
                    
                    for vi in [pi[0], pi[-1]]:
                        for vj in [pj[0], pj[-1]]:
                            if vi == vj:
                                # They share a vertex - try to merge
                                # Reverse if needed
                                if pi[-1] == vj:
                                    new_path = pi + pj[1:]
                                elif pi[0] == vj:
                                    new_path = [pi[0]] + pi[1:] + pj[1:] if pi[-1] != vj else pi + pj[1:]
                                    # Actually need to be careful
                                    new_path = pi + pj[1:]
                                elif pj[-1] == vi:
                                    new_path = pj + pi[1:]
                                elif pj[0] == vi:
                                    new_path = pj + pi[1:]
                                else:
                                    continue
                                
                                # Check if new_path is valid (all edges exist)
                                valid = True
                                for k in range(len(new_path) - 1):
                                    e = tuple(sorted((new_path[k], new_path[k+1])))
                                    if e not in edge_set:
                                        valid = False
                                        break
                                # Check if new_path is simple
                                if len(set(new_path)) != len(new_path):
                                    valid = False
                                
                                if valid:
                                    # Replace paths i and j with merged path
                                    paths[i] = new_path
                                    paths.pop(j)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return paths

    # Process each graph
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        edges = instance(*spec)
        
        if time.time() < deadline - 5:
            paths = _path_decomposition(edges, max_time=deadline - 5)
            # Local search to improve
            if time.time() < deadline - 3:
                paths = _local_search_improve(paths, edges, max_time=deadline - 3)
        else:
            paths = [[list(e)] for e in edges]
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
