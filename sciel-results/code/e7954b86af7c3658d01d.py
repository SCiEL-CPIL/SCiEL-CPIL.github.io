import time
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']

# Precompute arrival index (0-based) for each edge
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    key = (min(u, v), max(u, v))
    arrival[key] = idx

def get_arrival(u, v):
    key = (min(u, v), max(u, v))
    return arrival[key]

def check_connectivity(adj, n):
    """Check if the graph is connected using BFS."""
    if not adj[0]:
        return False
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    count = 1
    while queue:
        cur = queue.popleft()
        for nb in adj[cur]:
            if not visited[nb]:
                visited[nb] = True
                count += 1
                queue.append(nb)
    return count == n

def has_hamilton_cycle_dfs(max_arrival, time_limit, start_vertex=0):
    """
    Find a Hamiltonian cycle using only edges with arrival <= max_arrival
    using DFS with strong pruning. Returns the cycle or None.
    """
    # Build adjacency list for edges with arrival <= max_arrival
    adj = [set() for _ in range(n)]
    for idx in range(max_arrival + 1):
        u, v = edge_order[idx]
        adj[u].add(v)
        adj[v].add(u)
    
    # Check connectivity
    if not check_connectivity(adj, n):
        return None
    
    # Check if all vertices have degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    deadline = time.time() + time_limit
    visited = [False] * n
    path = [start_vertex]
    visited[start_vertex] = True
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            last = path[-1]
            if get_arrival(last, start_vertex) <= max_arrival:
                return path[:]
            return None
        
        current = path[-1]
        remaining = n - len(path) - 1
        
        # Get unvisited neighbors
        candidates = []
        for nb in adj[current]:
            if visited[nb]:
                continue
            # Count unvisited neighbors of nb for Warnsdorff heuristic
            unvisited_count = sum(1 for x in adj[nb] if not visited[x])
            candidates.append((unvisited_count, get_arrival(current, nb), nb))
        
        candidates.sort()
        
        for _, _, nb in candidates:
            # Pruning: if nb is the last unvisited vertex, it must connect to start
            if remaining == 1 and get_arrival(nb, start_vertex) > max_arrival:
                continue
            
            # Additional pruning: check if nb can still reach start through unvisited vertices
            # This is a simplified connectivity check
            if remaining > 1:
                # Quick check: nb must have at least one unvisited neighbor that isn't the current vertex
                has_unvisited_nb = any(not visited[x] and x != current for x in adj[nb])
                if not has_unvisited_nb:
                    continue
            
            visited[nb] = True
            path.append(nb)
            result = backtrack()
            if result is not None:
                return result
            path.pop()
            visited[nb] = False
        
        return None
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin before 90s kill
    
    # Default cycle: 0-1-2-...-23-0
    default_cycle = list(range(n))
    default_score = max(get_arrival(default_cycle[i], default_cycle[(i+1) % n]) + 1 for i in range(n))
    
    best_score = default_score
    best_cycle = default_cycle
    
    # Binary search on the threshold
    # First, find an upper bound by trying increasing values
    lo = 23  # Minimum possible (need at least n edges for a cycle)
    hi = 276  # Maximum possible
    
    # Quick check: can we find a cycle with hi?
    if time.time() < deadline - 10:
        test_cycle = has_hamilton_cycle_dfs(hi, 2.0)
        if test_cycle is not None:
            score = max(get_arrival(test_cycle[i], test_cycle[(i+1) % n]) + 1 for i in range(n))
            if score < best_score:
                best_score = score
                best_cycle = test_cycle
            hi = min(hi, score - 1)
    
    # Binary search
    while lo < hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        time_per_attempt = min(3.0, (deadline - 5 - time.time()) * 0.4)
        if time_per_attempt < 0.5:
            break
        
        cycle = has_hamilton_cycle_dfs(mid, time_per_attempt)
        
        if cycle is not None:
            # Verify the cycle
            if len(cycle) != n:
                continue
            score = max(get_arrival(cycle[i], cycle[(i+1) % n]) + 1 for i in range(n))
            if score <= mid + 1:
                if score < best_score:
                    best_score = score
                    best_cycle = cycle
                hi = mid - 1
            else:
                # Score is higher than expected, tighten hi
                hi = min(hi, score - 1)
        else:
            lo = mid + 1
    
    # Try to improve with the best found threshold
    if time.time() < deadline - 3 and best_score > 1:
        final_threshold = best_score - 1
        time_remaining = deadline - 3 - time.time()
        if time_remaining > 1.0:
            cycle = has_hamilton_cycle_dfs(final_threshold, time_remaining)
            if cycle is not None and len(cycle) == n:
                score = max(get_arrival(cycle[i], cycle[(i+1) % n]) + 1 for i in range(n))
                if score < best_score:
                    best_score = score
                    best_cycle = cycle
    
    return best_cycle
# EVOLVE-BLOCK-END
