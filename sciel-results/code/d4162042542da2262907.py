"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict, deque

def construct():
    """Return three lists of paths (vertex lists) for the three fixed instance graphs."""
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose edges into minimum number of edge-disjoint simple paths covering all edges."""
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        adj = defaultdict(list)
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((u, v))

        # Find odd-degree vertices
        odd_vertices = sorted([v for v in range(n) if len(adj[v]) % 2 == 1])
        target_paths = max(1, len(odd_vertices) // 2)

        # Use iterative greedy merging with multiple restarts
        best_paths = None
        best_count = float('inf')
        
        for restart in range(100):
            if time.time() > deadline - 10.0:
                break
                
            paths = _greedy_decompose_once(adj, edge_set, n, restart)
            count = len(paths)
            
            if count < best_count:
                best_count = count
                best_paths = paths
            
            if best_count <= target_paths:
                break
        
        if best_paths is None:
            best_paths = _greedy_decompose_once(adj, edge_set, n, 0)
        
        return best_paths

    def _greedy_decompose_once(adj, edge_set, n, seed):
        """Single greedy decomposition pass."""
        rng = random.Random(seed)
        
        # Start with each edge as its own path
        edge_list = list(edge_set)
        rng.shuffle(edge_list)
        paths = [[u, v] for u, v in edge_list]
        
        # Greedy merging: repeatedly merge paths that share endpoints
        # Prefer merges that result in longer paths
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Try all pairs, prefer merges that create longer paths
            best_merge = None
            best_len = 0
            best_i, best_j = -1, -1
            
            i = 0
            while i < len(paths) and time.time() < deadline - 5.0:
                j = i + 1
                while j < len(paths) and time.time() < deadline - 5.0:
                    result = _try_merge(paths[i], paths[j])
                    if result is not None:
                        if len(result) > best_len:
                            best_len = len(result)
                            best_merge = result
                            best_i, best_j = i, j
                    j += 1
                i += 1
            
            if best_merge is not None:
                paths[best_i] = best_merge
                paths.pop(best_j)
                changed = True
        
        return paths

    def _try_merge(p1, p2):
        """Try to merge two paths, return merged path or None."""
        candidates = []
        
        if p1[-1] == p2[0]:
            cand = p1 + p2[1:]
            if len(set(cand)) == len(cand):
                candidates.append(cand)
        
        if p1[-1] == p2[-1]:
            cand = p1 + list(reversed(p2[:-1]))
            if len(set(cand)) == len(cand):
                candidates.append(cand)
        
        if p1[0] == p2[0]:
            cand = list(reversed(p1)) + p2[1:]
            if len(set(cand)) == len(cand):
                candidates.append(cand)
        
        if p1[0] == p2[-1]:
            cand = list(reversed(p1)) + p2[:-1]
            if len(set(cand)) == len(cand):
                candidates.append(cand)
        
        if candidates:
            return candidates[0]
        return None

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    result = []
    for spec in specs:
        edges = instance(*spec)
        paths = decompose(edges)
        result.append(paths)
    
    return result
# EVOLVE-BLOCK-END
