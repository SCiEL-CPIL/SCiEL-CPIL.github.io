"""Find a Hamilton cycle with minimal max arrival index in the pinned edge list."""
import time
import sys
from functools import lru_cache

# Precompute edge arrival indices (1-based)
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = 24
# Build edge index map: (u, v) -> arrival index (1-based), with u < v
edge_idx = {}
for i, (u, v) in enumerate(P['edge_order']):
    if u > v:
        u, v = v, u
    edge_idx[(u, v)] = i + 1

# For quick lookup
def get_edge_idx(u, v):
    if u > v:
        u, v = v, u
    return edge_idx[(u, v)]

# Build adjacency list sorted by arrival index for each vertex
adj = [[] for _ in range(N)]
for i, (u, v) in enumerate(P['edge_order']):
    idx = i + 1
    adj[u].append((idx, v))
    adj[v].append((idx, u))
for v in range(N):
    adj[v].sort()  # sort by arrival index

# Precompute for each vertex, the list of (arrival_idx, neighbor) sorted by arrival_idx
# Also build a set for fast edge existence check at a given threshold
# For threshold K, edge (u,v) is available if edge_idx[(u,v)] <= K

def has_hamilton_cycle(K, time_limit):
    """Check if there's a Hamilton cycle using only edges with arrival index <= K.
    Uses DFS with bitmask DP and pruning. Returns the cycle if found, else None."""
    start_time = time.time()
    
    # Build adjacency for threshold K
    adjK = [[] for _ in range(N)]
    for v in range(N):
        for idx, nb in adj[v]:
            if idx <= K:
                adjK[v].append(nb)
            else:
                break  # sorted, so all subsequent are > K
    
    # Check if graph is 2-connected enough: each vertex must have degree >= 2
    for v in range(N):
        if len(adjK[v]) < 2:
            return None
    
    # Use DFS with bitmask. Fix vertex 0 as start and its first neighbor as second.
    # State: (current_vertex, visited_mask)
    # We want to find a cycle: 0 -> ... -> 0
    
    # To make it faster, use iterative DFS with pruning
    # visited_mask: bit i set means vertex i is visited
    # We'll use a recursive approach with memoization of (vertex, mask) -> dead_end
    
    sys.setrecursionlimit(100000)
    
    dead = {}  # (vertex, mask) -> True if dead end
    
    def dfs(v, mask, path):
        if time.time() - start_time > time_limit:
            return None
        if mask == (1 << N) - 1:
            # All visited, check if v connects back to 0
            if 0 in adjK[v]:
                return path
            return None
        if (v, mask) in dead:
            return None
        
        # Try neighbors in order of arrival index (prefer early edges)
        for nb in adjK[v]:
            if mask & (1 << nb):
                continue
            if nb == 0:
                # Can only go back to 0 if all vertices visited
                if mask != (1 << N) - 1:
                    continue
            new_mask = mask | (1 << nb)
            result = dfs(nb, new_mask, path + [nb])
            if result is not None:
                return result
            if (nb, new_mask) not in dead:
                dead[(nb, new_mask)] = True
        
        dead[(v, mask)] = True
        return None
    
    # Start from vertex 0
    # First move: go to some neighbor of 0
    for nb in adjK[0]:
        if nb == 0:
            continue
        mask = (1 << 0) | (1 << nb)
        path = [0, nb]
        result = dfs(nb, mask, path)
        if result is not None:
            return result
        if time.time() - start_time > time_limit:
            return None
    
    return None

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    # Binary search on K
    # Lower bound: at least 24 (need 24 edges for a cycle)
    # Upper bound: 276 (total edges)
    lo = 24
    hi = 276
    
    best_cycle = None
    best_K = hi + 1
    
    # First, try a greedy/randomized approach to get an initial upper bound
    # Greedy: build a cycle by always picking the earliest available edge
    def greedy_cycle():
        # Start at vertex 0, greedily pick next vertex with smallest edge index
        visited = [False] * N
        visited[0] = True
        path = [0]
        current = 0
        for _ in range(N - 1):
            best_nb = -1
            best_idx = float('inf')
            for idx, nb in adj[current]:
                if not visited[nb]:
                    if idx < best_idx:
                        best_idx = idx
                        best_nb = nb
            if best_nb == -1:
                return None
            visited[best_nb] = True
            path.append(best_nb)
            current = best_nb
        # Check closing edge
        if 0 in adjK_check(current):
            return path
        return None
    
    def adjK_check(v):
        return [nb for idx, nb in adj[v] if idx <= hi]
    
    # Try greedy
    gc = greedy_cycle()
    if gc is not None:
        max_idx = 0
        for i in range(len(gc)):
            u = gc[i]
            v = gc[(i+1) % len(gc)]
            max_idx = max(max_idx, get_edge_idx(u, v))
        best_K = max_idx
        best_cycle = gc
        hi = max_idx - 1  # we want to find something better
    
    # Binary search
    while lo <= hi and time.time() - start_time < deadline:
        mid = (lo + hi) // 2
        time_per_check = min(3.0, max(0.5, (deadline - (time.time() - start_time)) / 2))
        cycle = has_hamilton_cycle(mid, time_per_check)
        if cycle is not None:
            # Verify the cycle
            max_idx = 0
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                max_idx = max(max_idx, get_edge_idx(u, v))
            if max_idx <= mid:
                best_cycle = cycle
                best_K = max_idx
                hi = mid - 1
            else:
                # Shouldn't happen, but be safe
                hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is None:
        # Fallback: return identity permutation (score will be based on its max edge)
        return list(range(N))
    
    return best_cycle
