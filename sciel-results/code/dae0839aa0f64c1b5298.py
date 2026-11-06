import time
import random
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']

# Precompute arrival index (0-based) for each edge
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    key = (min(u, v), max(u, v))
    arrival[key] = idx  # 0-based index

def get_arrival(u, v):
    key = (min(u, v), max(u, v))
    return arrival[key]

def find_hamiltonian_cycle_dp():
    """
    Use Held-Karp style DP to find the Hamiltonian cycle that minimizes
    the maximum edge arrival index (bottleneck objective).
    
    State: (mask, last) where mask is a bitmask of visited vertices, last is the last vertex.
    Value: minimum possible maximum edge arrival for a path from 0 to 'last' visiting exactly 'mask'.
    
    We fix vertex 0 as the start/end to break rotational symmetry.
    """
    INF = float('inf')
    
    # dp[mask][last] = min bottleneck value for path 0 -> ... -> last visiting exactly mask
    # mask is an integer bitmask, last is 0..n-1
    # Initialize with INF
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    
    # Start at vertex 0
    dp[1 << 0][0] = 0
    
    # Iterate over all masks
    for mask in range(1 << n):
        if not (mask & 1):  # Must include vertex 0
            continue
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if dp[mask][last] == INF:
                continue
            
            # Try extending to next vertex
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                
                edge_cost = get_arrival(last, nxt)
                new_bottleneck = max(dp[mask][last], edge_cost)
                new_mask = mask | (1 << nxt)
                
                if new_bottleneck < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = new_bottleneck
                    parent[new_mask][nxt] = last
    
    # Find the best completion: path from 0 through all vertices, last vertex must connect back to 0
    full_mask = (1 << n) - 1
    best_bottleneck = INF
    best_last = -1
    
    for last in range(1, n):
        if dp[full_mask][last] == INF:
            continue
        close_cost = get_arrival(last, 0)
        total_bottleneck = max(dp[full_mask][last], close_cost)
        if total_bottleneck < best_bottleneck:
            best_bottleneck = total_bottleneck
            best_last = last
    
    if best_last == -1:
        return None, INF
    
    # Reconstruct the path
    mask = full_mask
    last = best_last
    path = [last]
    
    while parent[mask][last] != -1:
        last = parent[mask][last]
        mask ^= (1 << last)  # Remove last from mask... wait, this is wrong
    
    # Let me redo reconstruction properly
    # parent[new_mask][nxt] = last, so to go backwards:
    # From state (mask, last), we came from (mask_without_last, prev) where prev = parent[mask][last]
    
    mask = full_mask
    last = best_last
    path = [last]
    
    while mask != (1 << 0):
        prev = parent[mask][last]
        if prev == -1:
            break
        path.append(prev)
        mask ^= (1 << last)
        last = prev
    
    path.reverse()
    # path should be [0, ..., best_last]
    
    if len(path) != n or path[0] != 0:
        return None, INF
    
    # Verify the cycle
    score = max(get_arrival(path[i], path[(i+1) % n]) + 1 for i in range(n))
    
    return path, score

def find_hamiltonian_cycle_backtracking(max_arrival, time_limit, start_vertex=0):
    """
    Find a Hamiltonian cycle using only edges with arrival <= max_arrival.
    Returns the cycle as a list of vertices, or None if not found within time_limit.
    """
    # Build adjacency list for edges with arrival <= max_arrival
    adj = {i: [] for i in range(n)}
    for idx, (u, v) in enumerate(edge_order[:max_arrival + 1]):
        adj[u].append(v)
        adj[v].append(u)
    
    # Sort neighbors by arrival time for heuristic
    for v in adj:
        adj[v].sort(key=lambda x: get_arrival(v, x))
    
    # Check if all vertices have degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Backtracking with pruning
    visited = [False] * n
    path = [start_vertex]
    visited[start_vertex] = True
    
    deadline = time.time() + time_limit
    best_cycle = None
    
    def backtrack():
        nonlocal best_cycle
        if time.time() > deadline:
            return False
        if len(path) == n:
            # Check if last vertex connects to start
            last = path[-1]
            if get_arrival(last, start_vertex) <= max_arrival:
                best_cycle = path[:]
                return True
            return False
        
        current = path[-1]
        # Heuristic: try neighbors with lowest arrival first
        for nb in adj[current]:
            if visited[nb]:
                continue
            # Pruning: if nb is not start and it's the last unvisited vertex, 
            # we need it to connect back to start
            remaining = n - len(path) - 1
            if remaining == 1 and get_arrival(nb, start_vertex) > max_arrival:
                continue
            visited[nb] = True
            path.append(nb)
            if backtrack():
                return True
            path.pop()
            visited[nb] = False
        return False
    
    backtrack()
    return best_cycle

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for 90s limit
    
    # First, compute score of the default cycle 0-1-2-...-23-0
    default_cycle = list(range(n))
    default_score = max(get_arrival(default_cycle[i], default_cycle[(i+1) % n]) + 1 for i in range(n))
    
    best_score = default_score
    best_cycle = default_cycle
    
    # Try DP approach first
    try:
        dp_cycle, dp_score = find_hamiltonian_cycle_dp()
        if dp_cycle is not None and dp_score < best_score:
            best_score = dp_score
            best_cycle = dp_cycle
    except Exception:
        pass
    
    # If DP didn't find a better solution, try backtracking with increasing thresholds
    if best_score == default_score:
        for max_arrival in range(23, 276):
            if time.time() > deadline - 5:
                break
            
            remaining_time = deadline - 5 - time.time()
            if remaining_time <= 0:
                break
            
            time_per_attempt = min(1.0, remaining_time * 0.5)
            
            cycle = find_hamiltonian_cycle_backtracking(max_arrival, time_per_attempt)
            
            if cycle is not None:
                score = max(get_arrival(cycle[i], cycle[(i+1) % n]) + 1 for i in range(n))
                if score < best_score:
                    best_score = score
                    best_cycle = cycle
                break
    
    return best_cycle
# EVOLVE-BLOCK-END
