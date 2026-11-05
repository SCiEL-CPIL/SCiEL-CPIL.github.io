"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
TOTAL_EDGES = len(EDGE_ORDER)

# Build edge index: for each edge (u,v) with u<v, store its 1-based arrival index
edge_index = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    a, b = (u, v) if u < v else (v, u)
    edge_index[(a, b)] = i

def edge_score(u, v):
    """Get the 1-based arrival index of edge (u,v)."""
    a, b = (u, v) if u < v else (v, u)
    return edge_index[(a, b)]

def cycle_max_index(cycle):
    """Compute the max edge arrival index in a Hamilton cycle."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = edge_score(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def build_adjacency(K):
    """Build adjacency list using edges 1..K."""
    adj = [set() for _ in range(N)]
    for i in range(1, K + 1):
        u, v = EDGE_ORDER[i - 1]
        adj[u].add(v)
        adj[v].add(u)
    return adj

def find_hamilton_cycle_dfs(adj, deadline):
    """Find a Hamilton cycle using DFS with pruning."""
    start = 0
    full_mask = (1 << N) - 1
    path = [start]
    visited = 1 << start
    
    # Sort neighbors by degree (Warnsdorff heuristic) for better pruning
    def get_neighbors(v, mask):
        unvisited = [u for u in adj[v] if not (mask & (1 << u))]
        # Sort by degree in remaining graph
        unvisited.sort(key=lambda u: len([w for w in adj[u] if not (mask & (1 << w))]))
        return unvisited
    
    def dfs(v, mask):
        if time.time() > deadline:
            return False
        if mask == full_mask:
            if start in adj[v]:
                return True
            return False
        
        for u in get_neighbors(v, mask):
            path.append(u)
            if dfs(u, mask | (1 << u)):
                return True
            path.pop()
        return False
    
    if dfs(start, visited):
        return path[:]
    return None

def greedy_construct(K, deadline):
    """Greedy construction of a Hamilton cycle using edges 1..K."""
    adj = build_adjacency(K)
    
    # Check minimum degree
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # Use Warnsdorff's heuristic
    start = 0
    path = [start]
    visited = set([start])
    
    def get_next(v, visited_set):
        candidates = [u for u in adj[v] if u not in visited_set]
        if not candidates:
            return None
        # Warnsdorff: choose neighbor with fewest unvisited neighbors
        candidates.sort(key=lambda u: len([w for w in adj[u] if w not in visited_set]))
        return candidates[0]
    
    for _ in range(N - 1):
        v = path[-1]
        nxt = get_next(v, visited)
        if nxt is None:
            return None
        path.append(nxt)
        visited.add(nxt)
    
    # Check if we can close the cycle
    if path[-1] in adj[start]:
        return path
    return None

def two_opt_improve(cycle, deadline, max_iter=10000):
    """Improve a Hamilton cycle using 2-opt, minimizing max edge index."""
    n = len(cycle)
    current_max = cycle_max_index(cycle)
    best_cycle = cycle[:]
    best_max = current_max
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue  # Would be the same edge
                
                # Try reversing segment i+1..j
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                new_max = cycle_max_index(new_cycle)
                
                if new_max < best_max:
                    best_max = new_max
                    best_cycle = new_cycle[:]
                    improved = True
                    # Don't break, keep looking for more improvements
        
        if not improved:
            break
    
    return best_cycle, best_max

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    best_cycle = list(range(N))
    best_score = cycle_max_index(best_cycle)
    
    # Strategy 1: Try increasing K values and find a Hamilton cycle
    # Lower bound: need at least N edges, and each vertex needs degree >= 2
    # Try K from a reasonable lower bound upward
    
    # First, find a good lower bound by checking when min degree >= 2
    lo = N
    for K in range(N, TOTAL_EDGES + 1):
        adj = build_adjacency(K)
        if all(len(adj[i]) >= 2 for i in range(N)):
            lo = K
            break
    
    # Try to find cycles for increasing K values
    for K in range(lo, min(lo + 100, TOTAL_EDGES + 1)):
        if time.time() > deadline:
            break
        
        adj = build_adjacency(K)
        
        # Check minimum degree
        if any(len(adj[i]) < 2 for i in range(N)):
            continue
        
        # Try DFS with Warnsdorff
        cycle = find_hamilton_cycle_dfs(adj, deadline)
        if cycle is not None:
            score = cycle_max_index(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle[:]
                # Try to improve with 2-opt
                improved_cycle, improved_score = two_opt_improve(cycle, deadline, max_iter=5000)
                if improved_score < best_score:
                    best_score = improved_score
                    best_cycle = improved_cycle[:]
                break  # Found a cycle, stop searching for smaller K
    
    # Strategy 2: If we found a cycle, try to improve it further
    if best_score < TOTAL_EDGES:
        improved_cycle, improved_score = two_opt_improve(best_cycle, deadline, max_iter=20000)
        if improved_score < best_score:
            best_score = improved_score
            best_cycle = improved_cycle[:]
    
    # Strategy 3: Multiple random restarts with greedy construction
    rng = random.Random(614)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random starting vertex
        start = rng.randint(0, N - 1)
        
        # Use a random K between lo and lo+50
        K = rng.randint(lo, min(lo + 50, TOTAL_EDGES))
        adj = build_adjacency(K)
        
        if any(len(adj[i]) < 2 for i in range(N)):
            continue
        
        # Greedy with random tie-breaking
        path = [start]
        visited = set([start])
        
        def get_next_random(v, visited_set, rng):
            candidates = [u for u in adj[v] if u not in visited_set]
            if not candidates:
                return None
            # Random tie-breaking among top candidates
            candidates.sort(key=lambda u: len([w for w in adj[u] if w not in visited_set]))
            top_k = min(3, len(candidates))
            return rng.choice(candidates[:top_k])
        
        success = True
        for _ in range(N - 1):
            v = path[-1]
            nxt = get_next_random(v, visited, rng)
            if nxt is None:
                success = False
                break
            path.append(nxt)
            visited.add(nxt)
        
        if success and path[-1] in adj[start]:
            score = cycle_max_index(path)
            if score < best_score:
                best_score = score
                best_cycle = path[:]
                # Improve
                improved_cycle, improved_score = two_opt_improve(path, deadline, max_iter=5000)
                if improved_score < best_score:
                    best_score = improved_score
                    best_cycle = improved_cycle[:]
    
    return best_cycle

# EVOLVE-BLOCK-END
