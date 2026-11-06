"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def find_path_greedy(adj, start, used_edges, target_len=None):
        """Find a long simple path from start using Warnsdorff's heuristic."""
        path = [start]
        visited = {start}
        cur = start
        while True:
            cands = []
            for v in adj[cur]:
                ek = edge_key(cur, v)
                if ek not in used_edges and v not in visited:
                    # Count onward options
                    onward = 0
                    for w in adj[v]:
                        if w not in visited and edge_key(v, w) not in used_edges:
                            onward += 1
                    cands.append((onward, v))
            if not cands:
                break
            cands.sort()
            cur = cands[0][1]
            ek = edge_key(path[-1], cur)
            used_edges.add(ek)
            path.append(cur)
            visited.add(cur)
            if target_len and len(path) >= target_len:
                break
        return path

    def decompose_graph(edges):
        if not edges:
            return []
        adj = build_adj(edges)
        total = len(edges)
        
        # Compute odd-degree vertices for lower bound
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_count = sum(1 for d in deg.values() if d % 2 == 1)
        lower_bound = max(1, odd_count // 2)
        
        best_paths = None
        best_count = total + 1
        
        rng = random.Random(12345)
        vertices = sorted(adj.keys())
        
        # Multiple random restarts with different orderings
        for attempt in range(200):
            if time_left() < 2.0:
                break
            
            used_edges = set()
            paths = []
            
            # Randomize start order
            order = vertices[:]
            rng.shuffle(order)
            
            for start in order:
                if time_left() < 1.0:
                    break
                if len(used_edges) >= total:
                    break
                
                # Check if start has any unused incident edge
                has_unused = False
                for v in adj[start]:
                    if edge_key(start, v) not in used_edges:
                        has_unused = True
                        break
                if not has_unused:
                    continue
                
                path = find_path_greedy(adj, start, used_edges)
                if len(path) > 1:
                    paths.append(path)
            
            if len(used_edges) == total and len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
                if best_count <= lower_bound:
                    break
        
        if best_paths is None:
            # Fallback: single-edge paths
            best_paths = [[u, v] for u, v in edges]
            best_count = len(edges)
        
        # Local search: try to merge pairs of paths
        if best_count > lower_bound and time_left() > 5.0:
            merged = True
            while merged and time_left() > 3.0:
                merged = False
                for i in range(len(best_paths)):
                    if time_left() < 3.0:
                        break
                    for j in range(i + 1, len(best_paths)):
                        pi = best_paths[i]
                        pj = best_paths[j]
                        # Try to merge: end of pi connects to start of pj, etc.
                        # Check if pi[-1] adjacent to pj[0] and edge unused in both paths
                        merged_path = None
                        for a, b in [(pi[-1], pj[0]), (pi[0], pj[-1]), (pi[-1], pj[-1]), (pi[0], pj[0])]:
                            if a == b:
                                continue
                            ek = edge_key(a, b)
                            # Check edge not in either path
                            in_i = any(edge_key(pi[k], pi[k+1]) == ek for k in range(len(pi)-1))
                            in_j = any(edge_key(pj[k], pj[k+1]) == ek for k in range(len(pj)-1))
                            if in_i or in_j:
                                continue
                            # Check vertex not already in the other path
                            if b in set(pi) or a in set(pj):
                                continue
                            # Build merged path
                            if a == pi[-1] and b == pj[0]:
                                candidate = pi + pj
                            elif a == pi[0] and b == pj[-1]:
                                candidate = list(reversed(pj)) + pi
                            elif a == pi[-1] and b == pj[-1]:
                                candidate = pi + list(reversed(pj))
                            elif a == pi[0] and b == pj[0]:
                                candidate = list(reversed(pj)) + list(reversed(pi))
                            else:
                                continue
                            # Verify simplicity
                            if len(set(candidate)) == len(candidate):
                                merged_path = candidate
                                break
                        if merged_path:
                            new_paths = best_paths[:i] + best_paths[i+1:j] + best_paths[j+1:]
                            new_paths.append(merged_path)
                            best_paths = new_paths
                            best_count -= 1
                            merged = True
                            break
                    if merged:
                        break
        
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time_left() < 3.0:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
