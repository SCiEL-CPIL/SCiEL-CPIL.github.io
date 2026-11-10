"""Minimum edge-disjoint simple path cover via ILP-style set cover + local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    _start = time.time()
    _DEADLINE = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def validate(paths, edge_set):
        covered = set()
        for p in paths:
            if len(p) < 2:
                return False
            if len(set(p)) != len(p):
                return False
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set:
                    return False
                if key in covered:
                    return False
                covered.add(key)
        return covered == edge_set

    def find_all_simple_paths(edge_set, adj, max_len=None, time_budget=10.0):
        """Find all simple paths up to a certain length or time budget."""
        if max_len is None:
            max_len = len(edge_set) + 1
        
        paths = []
        start_time = time.time()
        
        # Try all starting vertices
        for start in sorted(adj.keys()):
            stack = [(start, [start], {start})]
            while stack:
                if time.time() - start_time > time_budget:
                    return paths
                
                node, path, visited = stack.pop()
                if len(path) >= 2:
                    paths.append(list(path))
                
                if len(path) >= max_len:
                    continue
                
                for nb in adj[node]:
                    if nb not in visited:
                        key = (node, nb) if node <= nb else (nb, node)
                        if key in edge_set:
                            new_visited = visited | {nb}
                            stack.append((nb, path + [nb], new_visited))
        
        return paths

    def greedy_set_cover(edges_to_cover, candidate_paths, adj):
        """Greedy set cover: repeatedly pick the path covering most uncovered edges."""
        remaining = set(edges_to_cover)
        selected = []
        used_edges = set()
        
        while remaining:
            if time.time() - _start > _DEADLINE - 5:
                break
            
            best_path = None
            best_count = 0
            
            for path in candidate_paths:
                path_edges = set()
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    if key in remaining:
                        path_edges.add(key)
                
                if len(path_edges) > best_count:
                    best_count = len(path_edges)
                    best_path = path
            
            if best_path is None or best_count == 0:
                # No path covers any remaining edge - take single edges
                for e in remaining:
                    selected.append([e[0], e[1]])
                remaining.clear()
                break
            
            selected.append(best_path)
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key in remaining:
                    remaining.remove(key)
        
        return selected

    def local_search_improve(paths, edge_set, adj, time_budget=10.0):
        """Try to reduce number of paths by merging or restructuring."""
        start_time = time.time()
        current = [list(p) for p in paths]
        
        def is_simple(p):
            return len(set(p)) == len(p)
        
        def path_edges(p):
            es = set()
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                es.add((u, v) if u <= v else (v, u))
            return es
        
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            n = len(current)
            
            # Try merging pairs
            for i in range(n):
                if time.time() - start_time > time_budget:
                    break
                for j in range(i + 1, n):
                    p1, p2 = current[i], current[j]
                    
                    # Try all 4 endpoint connections
                    candidates = []
                    for a_end, b_end in [(True, False), (True, True), (False, False), (False, True)]:
                        a = p1[-1] if a_end else p1[0]
                        b = p2[-1] if b_end else p2[0]
                        key = (a, b) if a <= b else (b, a)
                        if key in edge_set:
                            # Check edge not already used in either path
                            e1 = path_edges(p1)
                            e2 = path_edges(p2)
                            if key not in e1 and key not in e2:
                                # Build merged path
                                rp1 = p1 if a_end else list(reversed(p1))
                                rp2 = p2 if not b_end else list(reversed(p2))
                                merged = rp1 + rp2[1:]
                                if is_simple(merged):
                                    candidates.append(merged)
                    
                    if candidates:
                        merged = candidates[0]
                        new_paths = [current[k] for k in range(n) if k != i and k != j]
                        new_paths.append(merged)
                        current = new_paths
                        improved = True
                        break
                if improved:
                    break
            
            # Try 2-opt style swaps: replace two edges with two others
            if not improved:
                for i in range(len(current)):
                    if time.time() - start_time > time_budget:
                        break
                    p = current[i]
                    for a in range(len(p) - 1):
                        for b in range(a + 1, len(p) - 1):
                            # Try reversing segment p[a+1..b]
                            new_p = p[:a+1] + list(reversed(p[a+1:b+1])) + p[b+1:]
                            if is_simple(new_p):
                                # Check all edges exist
                                valid = True
                                for k in range(len(new_p) - 1):
                                    u, v = new_p[k], new_p[k+1]
                                    key = (u, v) if u <= v else (v, u)
                                    if key not in edge_set:
                                        valid = False
                                        break
                                if valid:
                                    # Check edge-disjointness with other paths
                                    all_edges = set()
                                    for pi in current:
                                        if pi is not p:
                                            all_edges |= path_edges(pi)
                                    all_edges |= path_edges(new_p)
                                    if all_edges == edge_set:
                                        current[i] = new_p
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
        
        return current

    def solve_one(edges):
        if not edges:
            return []
        
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))
        
        adj = build_adjacency(edges)
        
        # Strategy 1: Find candidate paths and use greedy set cover
        time_budget = min(30.0, max(5.0, _DEADLINE - (time.time() - _start) - 10.0))
        
        # Find paths of various lengths
        all_paths = find_all_simple_paths(edge_set, adj, max_len=15, time_budget=time_budget * 0.5)
        
        if not all_paths:
            # Fallback: each edge as a path
            return [[u, v] for (u, v) in sorted(edge_set)]
        
        # Greedy set cover
        paths = greedy_set_cover(edge_set, all_paths, adj)
        
        if not paths or not validate(paths, edge_set):
            # Fallback: greedy longest path
            remaining = set(edge_set)
            paths = []
            while remaining:
                if time.time() - _start > _DEADLINE - 5:
                    break
                # Find longest path in remaining
                rem_adj = defaultdict(list)
                for (u, v) in remaining:
                    rem_adj[u].append(v)
                    rem_adj[v].append(u)
                
                if not rem_adj:
                    break
                
                start = max(rem_adj.keys(), key=lambda x: len(rem_adj[x]))
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    neighbors = [n for n in rem_adj[current] if n not in visited]
                    if not neighbors:
                        break
                    best = min(neighbors, key=lambda nb: len([x for x in rem_adj[nb] if x not in visited]))
                    path.append(best)
                    visited.add(best)
                    current = best
                
                path_edges = set()
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    path_edges.add(key)
                
                if path_edges:
                    remaining -= path_edges
                    paths.append(path)
                else:
                    e = next(iter(remaining))
                    remaining.remove(e)
                    paths.append([e[0], e[1]])
        
        # Local search improvement
        ls_time = min(20.0, max(2.0, _DEADLINE - (time.time() - _start) - 5.0))
        try:
            improved = local_search_improve(paths, edge_set, adj, ls_time)
            if validate(improved, edge_set) and len(improved) <= len(paths):
                paths = improved
        except Exception:
            pass
        
        # Final validation
        if not validate(paths, edge_set):
            paths = [[u, v] for (u, v) in sorted(edge_set)]
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - _start
        remaining = _DEADLINE - elapsed
        
        if remaining < 5:
            edges = instance(*spec)
            edge_set = set()
            for (u, v) in edges:
                edge_set.add((u, v) if u <= v else (v, u))
            results.append([[u, v] for (u, v) in sorted(edge_set)])
            continue
        
        edges = instance(*spec)
        paths = solve_one(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
