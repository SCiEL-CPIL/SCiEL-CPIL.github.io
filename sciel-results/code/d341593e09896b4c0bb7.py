"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

_N = 24

# Precompute the edge order from the given data
_EDGE_DATA = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build arrival index
_EDGE_INDEX = {}
for i, (u, v) in enumerate(_EDGE_DATA):
    if u > v:
        u, v = v, u
    _EDGE_INDEX[(u, v)] = i + 1


def _build_adj(k):
    """Build adjacency list using first k edges."""
    adj = [set() for _ in range(_N)]
    for i in range(k):
        u, v = _EDGE_DATA[i]
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_feasible(adj):
    """Quick check: all vertices have degree >= 2 and graph is connected."""
    for i in range(_N):
        if len(adj[i]) < 2:
            return False
    # BFS connectivity
    visited = [False] * _N
    queue = deque([0])
    visited[0] = True
    count = 1
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == _N


def _find_hc(adj, deadline):
    """Find Hamiltonian cycle with Warnsdorff + pruning. Returns list or None."""
    n = _N
    start = 0
    path = [start]
    in_path = [False] * n
    in_path[start] = True
    
    # Precompute neighbor lists as sorted lists for speed
    adj_lists = [sorted(adj[i]) for i in range(n)]
    
    def dfs(current, depth):
        if time.time() > deadline:
            return None
        
        if depth == n:
            if start in adj[current]:
                return list(path)
            return None
        
        # Get unvisited neighbors
        candidates = [nb for nb in adj_lists[current] if not in_path[nb]]
        if not candidates:
            return None
        
        # Warnsdorff: sort by number of unvisited neighbors (fewest first)
        candidates.sort(key=lambda x: sum(1 for nb in adj_lists[x] if not in_path[nb]))
        
        for next_v in candidates:
            if depth == n - 1:
                # Last vertex must connect to start
                if start not in adj[next_v]:
                    continue
            
            in_path[next_v] = True
            path.append(next_v)
            
            # Forward checking: ensure no unvisited vertex becomes isolated
            # (each unvisited vertex must have at least one unvisited neighbor or connect to start)
            ok = True
            for v in range(n):
                if not in_path[v]:
                    has_unvisited_nb = False
                    for nb in adj_lists[v]:
                        if not in_path[nb]:
                            has_unvisited_nb = True
                            break
                    if not has_unvisited_nb:
                        # Must connect to start to be placeable as last
                        if start not in adj[v]:
                            ok = False
                            break
                    # Also: if v only connects to start, it must be the very last
                    # (i.e., only one unvisited vertex remains)
                    if not has_unvisited_nb and start in adj[v]:
                        unvisited_count = sum(1 for x in range(n) if not in_path[x])
                        if unvisited_count > 1:
                            # v can only be last, but there are others remaining
                            # This is OK if others can reach v, but risky. 
                            # For now, allow it but it'll fail later.
                            pass
            
            if ok:
                result = dfs(next_v, depth + 1)
                if result is not None:
                    return result
            
            path.pop()
            in_path[next_v] = False
        
        return None
    
    return dfs(start, 1)


def construct():
    start_time = time.time()
    hard_deadline = start_time + 85
    
    n_edges = len(_EDGE_DATA)
    
    # Binary search for the minimum K where a Hamilton cycle exists
    # Lower bound: _N (need at least N edges for a cycle)
    # Upper bound: n_edges
    
    lo = _N
    hi = n_edges
    best_cycle = None
    best_k = n_edges + 1
    
    # First, verify that a cycle exists at hi
    adj_hi = _build_adj(hi)
    if _is_feasible(adj_hi):
        tb = min(5.0, max(1.0, hard_deadline - time.time() - 3))
        c = _find_hc(adj_hi, time.time() + tb)
        if c is not None:
            best_cycle = c
            best_k = hi
    
    if best_cycle is None:
        # Fallback
        best_cycle = list(range(_N))
        return best_cycle
    
    # Binary search
    while lo <= hi and time.time() < hard_deadline - 3:
        mid = (lo + hi) // 2
        if mid < best_k:
            adj_mid = _build_adj(mid)
            if _is_feasible(adj_mid):
                tb = min(3.0, max(0.5, hard_deadline - time.time() - 2))
                c = _find_hc(adj_mid, time.time() + tb)
                if c is not None:
                    best_cycle = c
                    best_k = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            else:
                lo = mid + 1
        else:
            hi = mid - 1
    
    return best_cycle
# EVOLVE-BLOCK-END
