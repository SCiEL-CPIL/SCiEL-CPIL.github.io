"""Hamilton cycle with minimal bottleneck edge index on 24 vertices."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
TOTAL_EDGES = N * (N - 1) // 2  # 276

# Precompute edge index (1-based arrival) for every pair
edge_idx = [[0] * N for _ in range(N)]
for i, (u, v) in enumerate(P['edge_order']):
    edge_idx[u][v] = i + 1
    edge_idx[v][u] = i + 1

# Precompute neighbor lists sorted by arrival index for each vertex
neighbors = [[] for _ in range(N)]
for u in range(N):
    for v in range(N):
        if u != v:
            neighbors[u].append((edge_idx[u][v], v))
    neighbors[u].sort()

def exists_ham_cycle(K, time_limit):
    """Check if a Hamilton cycle exists using only edges with arrival index <= K."""
    # Build adjacency with only edges <= K
    adj = [[] for _ in range(N)]
    for u in range(N):
        for (idx, v) in neighbors[u]:
            if idx <= K:
                adj[u].append(v)
            else:
                break
    
    # Quick degree check: every vertex must have degree >= 2
    for u in range(N):
        if len(adj[u]) < 2:
            return False, None
    
    # Symmetry breaking: fix vertex 0 as start, and the first neighbor as the smallest-index neighbor
    # This breaks rotation and reflection symmetry
    start = 0
    if not adj[start]:
        return False, None
    
    # Try each possible second vertex (symmetry breaking on reflection)
    # We fix the first edge to go to the smallest neighbor, then try all second vertices
    # Actually for reflection breaking, we fix the first neighbor as the smallest one
    first_neighbor = min(adj[start])
    
    sys.setrecursionlimit(10000)
    
    def dfs(path, used, current):
        if time.time() > time_limit:
            return False, None
        
        if len(path) == N:
            # Check if we can close the cycle
            if start in adj[current]:
                return True, list(path)
            return False, None
        
        for v in adj[current]:
            if not used[v]:
                # Pruning: check if v can still connect back (degree consideration)
                # Simple pruning: if v is the last unvisited vertex, it must connect to start
                remaining = [x for x in range(N) if not used[x] and x != v]
                if len(remaining) == 1:
                    last = remaining[0]
                    if start not in adj[last] and last not in adj[start]:
                        continue
                    # Also v must connect to last
                    if last not in adj[v]:
                        continue
                
                used[v] = True
                path.append(v)
                result, cycle = dfs(path, used, v)
                if result:
                    return True, cycle
                path.pop()
                used[v] = False
    
    used = [False] * N
    used[start] = True
    used[first_neighbor] = True
    path = [start, first_neighbor]
    return dfs(path, used, first_neighbor)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    # Binary search on K
    # Lower bound: we need at least 24 edges in a cycle, so K must be at least the 24th smallest edge
    # Upper bound: TOTAL_EDGES
    lo = 1
    hi = TOTAL_EDGES
    
    # Find a good initial upper bound by trying to find any cycle
    best_K = None
    best_cycle = None
    
    # First, try to find a feasible K with binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        found, cycle = exists_ham_cycle(mid, deadline - 1)
        if found:
            best_K = mid
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: greedy construction
    # Try to build a cycle greedily
    visited = [False] * N
    cycle = [0]
    visited[0] = True
    current = 0
    
    for _ in range(N - 1):
        best_v = -1
        best_idx = float('inf')
        for v in adj[current] if 'adj' in dir() else neighbors[current]:
            if isinstance(v, tuple):
                v = v[1]
            if not visited[v] and v != 0:
                idx = edge_idx[current][v]
                if idx < best_idx:
                    best_idx = idx
                    best_v = v
        if best_v == -1:
            break
        visited[best_v] = True
        cycle.append(best_v)
        current = best_v
    
    if len(cycle) < N:
        # Just return identity
        return list(range(N))
    
    # Check if we can close
    if edge_idx[cycle[-1]][cycle[0]] <= TOTAL_EDGES:
        return cycle
    
    return list(range(N))
# EVOLVE-BLOCK-END
