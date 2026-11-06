"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

# Precompute edge position lookup as a 2D array for speed
_n = P['n']
_edge_pos_arr = np.zeros((_n, _n), dtype=np.int32)
for i, e in enumerate(P['edge_order']):
    u, v = e
    _edge_pos_arr[u, v] = i + 1
    _edge_pos_arr[v, u] = i + 1

def _cycle_score(cycle):
    """Compute the score of a Hamilton cycle: max edge position in the cycle."""
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        pos = _edge_pos_arr[u, v]
        if pos > max_pos:
            max_pos = pos
    return max_pos

def _build_adj(K):
    """Build adjacency list for edges with position <= K."""
    adj = [set() for _ in range(_n)]
    for i, e in enumerate(P['edge_order']):
        if i + 1 > K:
            break
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamilton_cycle(adj, n, time_limit=2.0):
    """Find a Hamilton cycle using backtracking with Warnsdorff heuristic."""
    start = 0
    deadline = time.time() + time_limit
    
    def backtrack(path, visited):
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            if start in adj[path[-1]]:
                return path[:]
            return None
        
        last = path[-1]
        # Get unvisited neighbors, sorted by degree (Warnsdorff)
        candidates = [v for v in adj[last] if not visited[v]]
        if not candidates:
            return None
        
        # Sort by number of unvisited neighbors (least first)
        candidates.sort(key=lambda v: sum(1 for u in adj[v] if not visited[u]))
        
        for v in candidates:
            # Pruning: last vertex must connect back to start
            if len(path) == n - 1 and start not in adj[v]:
                continue
            visited[v] = True
            path.append(v)
            result = backtrack(path, visited)
            if result is not None:
                return result
            path.pop()
            visited[v] = False
        return None
    
    visited = [False] * n
    visited[start] = True
    return backtrack([start], visited)

def _two_opt(cycle, time_limit=3.0):
    """Improve cycle using 2-opt, minimizing max edge position."""
    n = len(cycle)
    best = cycle[:]
    best_score = _cycle_score(best)
    deadline = time.time() + time_limit
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                u, v = best[i], best[i + 1]
                w, x = best[j], best[(j + 1) % n]
                
                # New edges would be (u,w) and (v,x)
                new_e1 = _edge_pos_arr[u, w]
                new_e2 = _edge_pos_arr[v, x]
                
                # Current edges being removed
                old_e1 = _edge_pos_arr[u, v]
                old_e2 = _edge_pos_arr[w, x]
                
                # Quick check: if both new edges are <= current max, try it
                if new_e1 <= best_score and new_e2 <= best_score:
                    new_cycle = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                    new_score = _cycle_score(new_cycle)
                    if new_score < best_score:
                        best = new_cycle
                        best_score = new_score
                        improved = True
    
    return best, best_score

def _greedy_cycle(K):
    """Greedy construction of a cycle using only edges with pos <= K."""
    n = _n
    adj = _build_adj(K)
    if any(len(adj[v]) < 2 for v in range(n)):
        return None
    
    # Use DFS with Warnsdorff
    return _find_hamilton_cycle(adj, n, time_limit=2.0)

def construct():
    n = _n
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin before 90s kill
    
    # Start with identity permutation
    best_cycle = list(range(n))
    best_score = _cycle_score(best_cycle)
    
    # Binary search for minimum threshold K such that G_K contains a Hamilton cycle
    lo, hi = n, best_score
    
    # First verify hi is feasible (it should be since we have a cycle)
    # Binary search
    while lo < hi and time.time() < deadline - 15.0:
        mid = (lo + hi) // 2
        adj = _build_adj(mid)
        
        # Quick check: all vertices degree >= 2
        if any(len(adj[v]) < 2 for v in range(n)):
            lo = mid + 1
            continue
        
        # Try to find Hamilton cycle
        cycle = _find_hamilton_cycle(adj, n, time_limit=2.0)
        if cycle is not None:
            score = _cycle_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
            hi = mid
        else:
            lo = mid + 1
    
    # Local improvement with 2-opt
    if time.time() < deadline - 5.0:
        improved_cycle, improved_score = _two_opt(best_cycle, time_limit=10.0)
        if improved_score < best_score:
            best_score = improved_score
            best_cycle = improved_cycle
    
    # Additional restarts with random perturbations if time allows
    if time.time() < deadline - 10.0:
        rng = random.Random(42)
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            # Perturb best cycle
            perturbed = best_cycle[:]
            for _ in range(3):
                i, j = rng.sample(range(n), 2)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            # Try 2-opt from perturbed
            cyc, sc = _two_opt(perturbed, time_limit=1.0)
            if sc < best_score:
                best_score = sc
                best_cycle = cyc
    
    return best_cycle
# EVOLVE-BLOCK-END
