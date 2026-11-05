"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_graph(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def path_edges(path):
        return [tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)]

    def is_simple_path(path, adj):
        if len(path) < 2:
            return False
        seen = set(path)
        if len(seen) != len(path):
            return False
        for i in range(len(path)-1):
            if path[i+1] not in adj[path[i]]:
                return False
        return True

    def initial_path_cover(adj, all_edges):
        # Use a greedy approach: repeatedly find long paths
        used_edges = set()
        paths = []
        remaining = set(all_edges)
        
        # Sort edges for determinism
        for edge in all_edges:
            if edge not in remaining:
                continue
            # Try to extend from this edge
            u, v = edge
            path = [u, v]
            used_edges.add(tuple(sorted((u, v))))
            remaining.discard(tuple(sorted((u, v))))
            
            # Try to extend forward
            extended = True
            while extended:
                extended = False
                last = path[-1]
                for nb in sorted(adj[last]):
                    e = tuple(sorted((last, nb)))
                    if e in remaining and nb not in path:
                        path.append(nb)
                        used_edges.add(e)
                        remaining.discard(e)
                        extended = True
                        break
            
            # Try to extend backward
            extended = True
            while extended:
                extended = False
                first = path[0]
                for nb in sorted(adj[first]):
                    e = tuple(sorted((first, nb)))
                    if e in remaining and nb not in path:
                        path.insert(0, nb)
                        used_edges.add(e)
                        remaining.discard(e)
                        extended = True
                        break
            
            if len(path) >= 2:
                paths.append(path)
        
        return paths

    def try_merge(paths, adj):
        """Try to merge two paths that share an endpoint."""
        n_paths = len(paths)
        merged = False
        for i in range(n_paths):
            for j in range(i+1, n_paths):
                if merged:
                    break
                p1, p2 = paths[i], paths[j]
                # Check if we can merge: p1's end connects to p2's start, etc.
                # Four possible merges:
                # 1. p1[-1] == p2[0] or edge p1[-1]-p2[0] exists and no vertex conflict
                # 2. p1[-1] == p2[-1] (reverse p2)
                # 3. p1[0] == p2[0] (reverse p1)
                # 4. p1[0] == p2[-1] (reverse both)
                
                candidates = []
                # p1 + p2
                if len(p1) >= 2 and len(p2) >= 2:
                    # Check edge p1[-1] to p2[0]
                    if p2[0] in adj[p1[-1]]:
                        new_path = p1 + p2[1:]
                        if is_simple_path(new_path, adj):
                            candidates.append(new_path)
                    # Check edge p1[-1] to p2[-1] (reverse p2)
                    if p2[-1] in adj[p1[-1]]:
                        new_path = p1 + list(reversed(p2))[1:]
                        if is_simple_path(new_path, adj):
                            candidates.append(new_path)
                    # Check edge p1[0] to p2[0] (reverse p1)
                    if p2[0] in adj[p1[0]]:
                        new_path = list(reversed(p1))[1:] + p2
                        if is_simple_path(new_path, adj):
                            candidates.append(new_path)
                    # Check edge p1[0] to p2[-1] (reverse p1 and p2)
                    if p2[-1] in adj[p1[0]]:
                        new_path = list(reversed(p1))[1:] + list(reversed(p2))
                        if is_simple_path(new_path, adj):
                            candidates.append(new_path)
                
                if candidates:
                    # Pick the longest candidate
                    best = max(candidates, key=len)
                    paths[i] = best
                    paths.pop(j)
                    merged = True
                    break
            
            if merged:
                break
        
        return merged

    def local_search(paths, adj, all_edges, max_iters=200):
        """Local search: try merging, and if stuck, try splitting and re-merging."""
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            progress = try_merge(paths, adj)
            if not progress:
                break
        return paths

    def refine_with_random_restarts(paths, adj, all_edges, rng, n_restarts=50):
        """Try random perturbations and re-optimize."""
        best = paths
        best_len = len(paths)
        
        for _ in range(n_restarts):
            if time.time() > DEADLINE:
                break
            
            # Start from current best, add some random noise
            current = [list(p) for p in best]
            
            # Randomly split a path and try to re-merge differently
            if len(current) > 1:
                idx = rng.randrange(len(current))
                p = current[idx]
                if len(p) > 2:
                    # Split at a random internal point
                    split_at = rng.randrange(1, len(p)-1)
                    p1 = p[:split_at+1]
                    p2 = p[split_at:]
                    current[idx:idx+1] = [p1, p2]
            
            # Re-optimize
            current = local_search(current, adj, all_edges, max_iters=50)
            
            if len(current) < best_len:
                best = current
                best_len = len(current)
        
        return best

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            # Fallback: return remaining as single-edge paths
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        adj = build_graph(edges)
        
        # Initial path cover
        paths = initial_path_cover(adj, edges)
        
        # Local search
        paths = local_search(paths, adj, edges, max_iters=500)
        
        # Random restarts for further improvement
        rng = random.Random(42)
        paths = refine_with_random_restarts(paths, adj, edges, rng, n_restarts=100)
        
        # Final merge pass
        paths = local_search(paths, adj, edges, max_iters=200)
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
