"""Find a Hamilton cycle minimizing max edge arrival index."""
import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
TOTAL_EDGES = N * (N - 1) // 2  # 276

# Build edge index lookup
edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    key = (min(u, v), max(u, v))
    edge_idx[key] = i

# Build sorted adjacency: for each vertex, list of (neighbor, edge_index) sorted by index
adj_sorted = [[] for _ in range(N)]
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    adj_sorted[u].append((v, i))
    adj_sorted[v].append((u, i))
for v in range(N):
    adj_sorted[v].sort(key=lambda x: x[1])

def get_edge_index(u, v):
    return edge_idx[(min(u, v), max(u, v))]

def cycle_score(cycle):
    """Compute the max edge index for a given Hamilton cycle."""
    max_idx = 0
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = get_edge_index(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def build_masks(K):
    """Build bitmask adjacency for edges with index <= K."""
    masks = [0] * N
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        if i > K:
            break
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def is_connected(masks):
    """Check if graph defined by masks is connected."""
    visited = [False] * N
    queue = []
    start = -1
    for i in range(N):
        if masks[i] != 0:
            start = i
            break
    if start == -1:
        return False
    queue.append(start)
    visited[start] = True
    count = 1
    while queue:
        u = queue.pop(0)
        mask = masks[u]
        while mask:
            lowbit = mask & -mask
            v = lowbit.bit_length() - 1
            mask ^= lowbit
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == N

def find_cycle_dfs(K, time_limit, start_vertex=0):
    """Find a Hamilton cycle using DFS with Warnsdorff heuristic on edges <= K."""
    start_time = time.time()
    deadline = start_time + time_limit
    sys.setrecursionlimit(10000)
    
    masks = build_masks(K)
    
    # Check connectivity
    if not is_connected(masks):
        return None
    
    # Check minimum degree
    for v in range(N):
        if masks[v].bit_count() < 2:
            return None
    
    start = start_vertex
    full_mask = (1 << N) - 1
    visited_mask = 1 << start
    path = [start]
    
    def dfs(v):
        if time.time() > deadline:
            return False
        if len(path) == N:
            # Check closing edge
            if masks[v] & (1 << start):
                return True
            return False
        
        # Get unvisited neighbors
        unvisited = masks[v] & ~visited_mask
        
        if unvisited == 0:
            return False
        
        # Warnsdorff: sort by number of unvisited neighbors (fewest first)
        neighbors = []
        mask = unvisited
        while mask:
            lowbit = mask & -mask
            u = lowbit.bit_length() - 1
            mask ^= lowbit
            # Count unvisited neighbors of u
            u_unvisited = masks[u] & ~visited_mask
            # Remove current v from count since it's being visited
            u_unvisited &= ~(1 << v)
            neighbors.append((u, u_unvisited.bit_count()))
        
        neighbors.sort(key=lambda x: x[1])
        
        for u, _ in neighbors:
            visited_mask |= (1 << u)
            path.append(u)
            if dfs(u):
                return True
            path.pop()
            visited_mask &= ~(1 << u)
        
        return False
    
    if dfs(start):
        return list(path)
    return None

def find_cycle_greedy(K, time_limit, start_vertex=0):
    """Find a Hamilton cycle using greedy construction with backtracking."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    masks = build_masks(K)
    
    if not is_connected(masks):
        return None
    
    for v in range(N):
        if masks[v].bit_count() < 2:
            return None
    
    start = start_vertex
    visited = [False] * N
    visited[start] = True
    path = [start]
    
    # Greedy with limited backtracking
    def greedy_dfs(v, depth):
        if time.time() > deadline:
            return False
        if depth == N:
            if masks[v] & (1 << start):
                return True
            return False
        
        # Get unvisited neighbors sorted by edge index (prefer early edges)
        unvisited = masks[v] & ~((1 << N) - 1)  # placeholder
        # Actually compute unvisited neighbors
        unvisited_mask = 0
        for u in range(N):
            if not visited[u] and (masks[v] & (1 << u)):
                unvisited_mask |= (1 << u)
        
        if unvisited_mask == 0:
            return False
        
        # Sort by edge index
        neighbors = []
        mask = unvisited_mask
        while mask:
            lowbit = mask & -mask
            u = lowbit.bit_length() - 1
            mask ^= lowbit
            idx = get_edge_index(v, u)
            neighbors.append((idx, u))
        
        neighbors.sort()
        
        for idx, u in neighbors:
            visited[u] = True
            path.append(u)
            if greedy_dfs(u, depth + 1):
                return True
            path.pop()
            visited[u] = False
        return False
    
    if greedy_dfs(start, 1):
        return list(path)
    return None

def best_improvement_2opt(cycle, time_limit):
    """Apply best-improvement 2-opt to reduce the max edge index."""
    start_time = time.time()
    deadline = start_time + time_limit
    current_cycle = list(cycle)
    current_score = cycle_score(current_cycle)
    n = len(current_cycle)
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        best_new_score = current_score
        best_new_cycle = None
        best_i, best_j = -1, -1
        
        # Try all 2-opt moves
        for i in range(n):
            if (time.time() - start_time) > deadline:
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # This would just reverse the whole cycle
                
                # 2-opt: reverse segment from i+1 to j
                # Edges affected: (cycle[i], cycle[i+1]) and (cycle[j], cycle[(j+1)%n])
                # New edges: (cycle[i], cycle[j]) and (cycle[i+1], cycle[(j+1)%n])
                a = current_cycle[i]
                b = current_cycle[i + 1]
                c = current_cycle[j]
                d = current_cycle[(j + 1) % n]
                
                # Check if new edges are valid (not self-loops)
                if a == c or b == d:
                    continue
                
                new_idx1 = get_edge_index(a, c)
                new_idx2 = get_edge_index(b, d)
                
                # Compute new max score
                # The old edges (a,b) and (c,d) are removed
                # We need to recompute the full score, but we can approximate
                # by checking if the new edges are both <= current_score
                # and if they reduce the max
                
                if new_idx1 <= current_score and new_idx2 <= current_score:
                    # Create new cycle
                    new_cycle = current_cycle[:i+1] + current_cycle[i+1:j+1][::-1] + current_cycle[j+1:]
                    new_score = cycle_score(new_cycle)
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_cycle = new_cycle
                        best_i, best_j = i, j
        
        if best_new_cycle is not None:
            current_cycle = best_new_cycle
            current_score = best_new_score
            improved = True
    
    return current_cycle

def construct():
    start_time = time.time()
    overall_deadline = start_time + 85  # Safety margin before 90s
    
    # Binary search on the maximum edge index
    lo = N  # minimum possible (24 edges needed)
    hi = TOTAL_EDGES
    
    best_cycle = None
    best_max_edge = TOTAL_EDGES + 1
    
    # First, find any valid cycle with all edges
    initial_cycle = find_cycle_dfs(TOTAL_EDGES, 5.0)
    if initial_cycle is None:
        initial_cycle = find_cycle_greedy(TOTAL_EDGES, 5.0)
    
    if initial_cycle is not None:
        best_cycle = initial_cycle
        best_max_edge = cycle_score(initial_cycle)
        hi = best_max_edge - 1
    
    # Binary search to find the minimum max_edge
    while lo <= hi and (time.time() - start_time) < 80:
        mid = (lo + hi) // 2
        
        # Allocate time proportional to remaining budget
        remaining = 80 - (time.time() - start_time)
        time_for_search = min(remaining * 0.7, 10.0)  # max 10s per search
        if time_for_search < 1:
            break
        
        # Try multiple starting vertices for better coverage
        found = False
        for start_v in range(N):
            if time.time() - start_v > 80 or time.time() - start_time > 80:
                break
            time_per_start = max(0.1, (time_for_search - (time.time() - start_time)) / max(1, N - start_v))
            if time_per_start < 0.05:
                break
            
            cycle = find_cycle_dfs(mid, time_per_start, start_vertex=start_v)
            if cycle is None:
                cycle = find_cycle_greedy(mid, time_per_start, start_vertex=start_v)
            
            if cycle is not None:
                score = cycle_score(cycle)
                if score <= mid:
                    best_cycle = cycle
                    best_max_edge = mid
                    hi = mid - 1
                    found = True
                    break
        
        if not found:
            lo = mid + 1
    
    # If we found a cycle, try to polish it with best-improvement 2-opt
    if best_cycle is not None:
        remaining_time = 85 - (time.time() - start_time)
        if remaining_time > 2:
            polished = best_improvement_2opt(best_cycle, remaining_time - 1)
            if polished is not None:
                polished_score = cycle_score(polished)
                if polished_score < best_max_edge:
                    best_cycle = polished
                    best_max_edge = polished_score
    
    if best_cycle is None:
        # Fallback: return identity permutation
        return list(range(N))
    
    return best_cycle
# EVOLVE-BLOCK-END
