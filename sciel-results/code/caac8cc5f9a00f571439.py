import time
import random
import numpy as np
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def greedy_maximal_paths(edges, seed_offset=0, max_attempts=20):
        """Extract maximal simple paths greedily, trying multiple randomizations."""
        rng = random.Random(seed_offset)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_paths = None
        best_count = float('inf')
        
        for attempt in range(max_attempts):
            remaining = set(edges)
            paths = []
            
            while remaining:
                # Pick a random edge to start
                if not remaining:
                    break
                start_edge = list(rng.sample(remaining, min(1, len(remaining))))[0]
                u, v = start_edge
                
                # Try both directions
                best_path = None
                for direction in [(u, v), (v, u)]:
                    path = [direction[0], direction[1]]
                    used_verts = set(path)
                    used_edges = {start_edge}
                    remaining.discard(start_edge)
                    
                    # Extend from both ends
                    extended = True
                    while extended:
                        extended = False
                        # Try extending from end
                        end = path[-1]
                        neighbors = [w for w in adj[end] if (min(end,w),max(end,w)) in remaining and w not in used_verts]
                        if neighbors:
                            # Choose neighbor that minimizes dead ends
                            best_next = min(neighbors, key=lambda w: sum(1 for x in adj[w] if (min(w,x),max(w,x)) in remaining and x not in used_verts and x != path[-2]))
                            path.append(best_next)
                            used_verts.add(best_next)
                            remaining.discard((min(end,best_next),max(end,best_next)))
                            extended = True
                        
                        # Try extending from start
                        if not extended:
                            start = path[0]
                            neighbors = [w for w in adj[start] if (min(start,w),max(start,w)) in remaining and w not in used_verts]
                            if neighbors:
                                best_next = min(neighbors, key=lambda w: sum(1 for x in adj[w] if (min(w,x),max(w,x)) in remaining and x not in used_verts and x != path[1]))
                                path.insert(0, best_next)
                                used_verts.add(best_next)
                                remaining.discard((min(start,best_next),max(start,best_next)))
                                extended = True
                    
                    if best_path is None or len(path) > len(best_path):
                        best_path = path
                
                if best_path:
                    paths.append(best_path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
            
            if best_count <= 1:
                break
        
        return best_paths

    def try_merge_paths(paths):
        """Try to merge two paths that share exactly one endpoint."""
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                p1, p2 = paths[i], paths[j]
                if len(p1) < 2 or len(p2) < 2:
                    continue
                set1, set2 = set(p1), set(p2)
                common = set1 & set2
                if len(common) != 1:
                    continue
                shared = list(common)[0]
                ep1 = (p1[0] == shared or p1[-1] == shared)
                ep2 = (p2[0] == shared or p2[-1] == shared)
                if not (ep1 and ep2):
                    continue
                
                # Try all 4 combinations
                for rev1 in [False, True]:
                    for rev2 in [False, True]:
                        a = p1[::-1] if rev1 else p1
                        b = p2[::-1] if rev2 else p2
                        if a[-1] == b[0]:
                            new_p = a + b[1:]
                            if len(new_p) == len(set(new_p)):
                                new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                                new_paths.append(new_p)
                                return new_paths
        return None

    def local_search(paths, deadline):
        """Iteratively merge paths until no more merges possible."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_paths = try_merge_paths(paths)
            if new_paths is not None:
                paths = new_paths
                changed = True
        return paths

    def verify(paths, edges):
        """Verify paths are valid: simple, use graph edges, edge-disjoint, cover all edges."""
        edge_set = set(edges)
        used = set()
        for p in paths:
            if len(p) < 2:
                continue
            if len(p) != len(set(p)):
                return False
            for i in range(len(p)-1):
                e = (min(p[i], p[i+1]), max(p[i], p[i+1]))
                if e not in edge_set or e in used:
                    return False
                used.add(e)
        return used == edge_set

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        best_paths = None
        best_count = float('inf')
        
        # Run greedy with multiple seeds
        for seed_offset in range(30):
            if time.time() > DEADLINE - 5:
                break
            paths = greedy_maximal_paths(edges, seed_offset, max_attempts=5)
            if paths is not None:
                paths = local_search(paths, DEADLINE - 5)
                if verify(paths, edges) and len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = paths
        
        if best_paths is None:
            # Fallback: simple edge-by-edge
            best_paths = [list(e) for e in edges]
        
        result.append(best_paths)
    
    return result
