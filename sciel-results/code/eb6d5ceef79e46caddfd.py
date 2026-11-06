"""Find Hamilton cycle with minimal max edge arrival index."""
import time
import sys
from collections import deque

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position lookup
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[frozenset((u, v))] = i


def build_adjacency(K):
    """Build adjacency list using only edges with position <= K, sorted by position."""
    adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        if i > K:
            break
        adj[u].append((v, i))
        adj[v].append((u, i))
    for v in range(N):
        adj[v].sort(key=lambda x: x[1])
    return adj


def check_connectivity(adj):
    """Check if the graph is connected."""
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        v = stack.pop()
        for nb, _ in adj[v]:
            if not visited[nb]:
                visited[nb] = True
                count += 1
                stack.append(nb)
    return count == N


def find_hamilton_cycle_dfs(adj, start, time_limit, max_nodes=None):
    """
    DFS with strong pruning to find a Hamilton cycle starting and ending at 'start'.
    Uses:
    - Degree constraint: each vertex in cycle must have degree 2
    - Connectivity: remaining unvisited vertices must be connected to current path
    - Warnsdorff: try neighbors with fewest unvisited neighbors first
    """
    sys.setrecursionlimit(10000)
    full_mask = (1 << N) - 1
    start_bit = 1 << start
    
    # Precompute neighbor lists as simple lists for speed
    neighbors = [sorted([nb for nb, _ in adj[v]]) for v in range(N)]
    
    # Check that start has at least 2 neighbors
    if len(adj[start]) < 2:
        return None
    
    # Check all vertices have degree >= 2 (necessary for Hamilton cycle)
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    # Check connectivity
    if not check_connectivity(adj):
        return None
    
    # For Warnsdorff, we need to count unvisited neighbors dynamically
    # We'll track the path and visited mask
    
    path = [start]
    visited_mask = start_bit
    current = start
    
    # We need the last two vertices to check degree constraints
    # Actually, let's use a simpler approach: track which vertices have been "entered"
    # and check at the end
    
    # Better approach: use backtracking with careful pruning
    # Track: path, visited_mask
    # At each step, we can only add a vertex if:
    # 1. It's not visited
    # 2. If it's not the last step, it must have at least one unvisited neighbor (or be 'start' if we're closing)
    # 3. If we're about to close, current must connect to start
    
    # Let's use iterative DFS with explicit stack for better time control
    # State: (current, visited_mask, path)
    # But path is expensive to copy. Let's use recursion with time checks.
    
    def dfs(v, visited_mask, path):
        if time.time() > time_limit:
            return None
        if max_nodes and len(path) > max_nodes:
            return None
            
        if len(path) == N:
            # Check if v connects back to start
            for nb, _ in adj[v]:
                if nb == start:
                    return path[:]
            return None
        
        # Get unvisited neighbors
        unvisited_neighbors = []
        for nb, pos in adj[v]:
            if not (visited_mask & (1 << nb)):
                unvisited_neighbors.append(nb)
        
        if not unvisited_neighbors:
            return None
        
        # Warnsdorff: sort by number of unvisited neighbors (fewest first)
        def warnsdorff_key(nb):
            count = 0
            for nb2, _ in adj[nb]:
                if not (visited_mask & (1 << nb2)):
                    count += 1
            # Also consider if nb can connect back to start if it's the last step
            return count
        
        unvisited_neighbors.sort(key=warnsdorff_key)
        
        for nb in unvisited_neighbors:
            # Pruning: if nb is not start and nb != v, check that nb has enough unvisited neighbors
            # to complete the path (unless it's the last vertex before closing)
            new_visited = visited_mask | (1 << nb)
            remaining = N - len(path) - 1  # vertices left after adding nb
            
            if remaining > 0:
                # nb must have at least one unvisited neighbor (to continue the path)
                # OR nb must be able to connect to start if it's the last step
                has_unvisited = False
                for nb2, _ in adj[nb]:
                    if not (new_visited & (1 << nb2)):
                        has_unvisited = True
                        break
                if not has_unvisited:
                    # Check if nb can connect to start (only valid if this is the last step)
                    if remaining == 0:
                        # nb is the last vertex, must connect to start
                        can_close = False
                        for nb2, _ in adj[nb]:
                            if nb2 == start:
                                can_close = True
                                break
                        if not can_close:
                            continue
                    else:
                        continue
            
            result = dfs(nb, new_visited, path + [nb])
            if result is not None:
                return result
        
        return None
    
    return dfs(start, start_bit, [start])


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s kill)
    
    # Strategy: binary search for minimum K, but with multiple DFS attempts per K
    # For each K, try several starting vertices before giving up
    
    lo, hi = 24, len(EDGE_ORDER)
    best_cycle = None
    best_K = None
    
    while lo <= hi:
        if time.time() > deadline - 10:
            break
        mid = (lo + hi) // 2
        
        # Try to find a Hamilton cycle in G_mid
        adj = build_adjacency(mid)
        
        # Quick checks
        if not check_connectivity(adj):
            lo = mid + 1
            continue
        
        # Check all vertices have degree >= 2
        degree_ok = True
        for v in range(N):
            if len(adj[v]) < 2:
                degree_ok = False
                break
        if not degree_ok:
            lo = mid + 1
            continue
        
        # Try DFS with multiple starting vertices
        found = None
        # Try vertex 0 first, then others
        starts = [0] + list(range(1, N))
        time_per_start = min(3.0, max(0.5, (deadline - 10 - time.time()) / len(starts)))
        
        for start_v in starts:
            if time.time() > deadline - 10:
                break
            # Check start_v has degree >= 2
            if len(adj[start_v]) < 2:
                continue
            
            t_start = time.time()
            cycle = find_hamilton_cycle_dfs(adj, start_v, t_start + time_per_start)
            if cycle is not None:
                found = cycle
                break
        
        if found is not None:
            best_cycle = found
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: try larger K values with more generous time
    for K in range(100, len(EDGE_ORDER) + 1, 3):
        if time.time() > deadline - 5:
            break
        adj = build_adjacency(K)
        if not check_connectivity(adj):
            continue
        degree_ok = all(len(adj[v]) >= 2 for v in range(N))
        if not degree_ok:
            continue
        
        starts = [0, 1, 2, 3, 4]
        for start_v in starts:
            if time.time() > deadline - 5:
                break
            if len(adj[start_v]) < 2:
                continue
            cycle = find_hamilton_cycle_dfs(adj, start_v, time.time() + 5.0)
            if cycle is not None:
                best_cycle = cycle
                break
        if best_cycle is not None:
            break
    
    if best_cycle is not None:
        return best_cycle
    
    # Last resort: identity permutation
    return list(range(N))


# Test
if __name__ == "__main__":
    t0 = time.time()
    cycle = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Cycle: {cycle}")
    if len(cycle) == N and len(set(cycle)) == N:
        max_pos = 0
        for i in range(N):
            u = cycle[i]
            v = cycle[(i+1) % N]
            pos = edge_pos[frozenset((u, v))]
            max_pos = max(max_pos, pos)
            print(f"  {u}-{v}: pos {pos}")
        print(f"Score (max edge pos): {max_pos}")
    else:
        print("INVALID CYCLE")
