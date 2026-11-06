"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Precompute edge position (1-based index)
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    key = (min(u, v), max(u, v))
    edge_pos[key] = i + 1

def _connectivity_check(adj):
    """Check if graph is connected using BFS."""
    visited = [False] * N
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == N

def find_hamilton_cycle_k(K, deadline, start_vertex=0):
    """Try to find a Hamilton cycle using only edges with position <= K."""
    if time.time() > deadline:
        return None
    
    # Build adjacency from first K edges
    adj = [set() for _ in range(N)]
    for i in range(K):
        u, v = EDGE_ORDER[i]
        adj[u].add(v)
        adj[v].add(u)
    
    # Check each vertex has degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    # Connectivity check
    if not _connectivity_check(adj):
        return None
    
    # Warnsdorff's heuristic backtracking DFS
    path = [start_vertex]
    visited = [False] * N
    visited[start_vertex] = True
    
    def dfs():
        if time.time() > deadline:
            return None
        if len(path) == N:
            # Check if last vertex connects to start
            last = path[-1]
            if start_vertex in adj[last]:
                return path[:]
            return None
        last = path[-1]
        
        # Warnsdorff: prefer neighbors with fewer unvisited neighbors
        # Also ensure the last vertex can connect back to start
        candidates = []
        for nxt in adj[last]:
            if not visited[nxt]:
                if len(path) == N - 1:
                    # Last vertex must connect to start
                    if start_vertex in adj[nxt]:
                        candidates.append(nxt)
                else:
                    # Count unvisited neighbors of nxt (excluding last)
                    unvisited_count = sum(1 for nn in adj[nxt] if not visited[nn] and nn != last)
                    candidates.append((unvisited_count, nxt))
        
        if not candidates:
            return None
        
        if len(path) == N - 1:
            # Sort by edge position for determinism
            candidates.sort(key=lambda x: edge_pos.get((min(last, x), max(last, x)), 999))
        else:
            # Sort by Warnsdorff (fewest unvisited neighbors first), then edge position
            candidates.sort(key=lambda x: (x[0], edge_pos.get((min(last, x[1]), max(last, x[1])), 999)))
        
        for item in candidates:
            nxt = item if len(path) == N - 1 else item[1]
            path.append(nxt)
            visited[nxt] = True
            result = dfs()
            if result is not None:
                return result
            path.pop()
            visited[nxt] = False
        return None
    
    return dfs()

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin (evaluator kills at 90s)
    
    # Fallback: identity permutation
    best_cycle = list(range(N))
    
    # Binary search on K (the prefix index)
    # Lower bound: at least N edges needed for a cycle
    # Upper bound: total edges = 276
    lo = N
    hi = 276
    
    # First, find the smallest K where a Hamilton cycle exists
    # Use binary search since existence is monotonic
    best_found_K = None
    best_found_cycle = None
    
    # Try binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        found = False
        
        # Try multiple start vertices
        for sv in range(N):
            if time.time() > deadline:
                break
            cycle = find_hamilton_cycle_k(mid, deadline, start_vertex=sv)
            if cycle is not None:
                # Verify
                max_pos = 0
                for i in range(N):
                    u = cycle[i]
                    v = cycle[(i + 1) % N]
                    key = (min(u, v), max(u, v))
                    pos = edge_pos.get(key, 999)
                    max_pos = max(max_pos, pos)
                if max_pos <= mid:
                    found = True
                    best_found_K = max_pos
                    best_found_cycle = cycle
                    break
        
        if found:
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_found_cycle is not None:
        return best_found_cycle
    
    # Fallback if binary search didn't find anything
    # Try greedy: build cycle from earliest edges
    return best_cycle
# EVOLVE-BLOCK-END
