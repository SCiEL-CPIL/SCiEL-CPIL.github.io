import time
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']
num_edges = len(edge_order)

# Precompute edge arrival indices
edge_index = {}
for i, (u, v) in enumerate(edge_order):
    edge_index[(min(u,v), max(u,v))] = i + 1  # 1-based

def get_edge_idx(u, v):
    if u > v:
        u, v = v, u
    return edge_index.get((u, v), num_edges + 1)

def has_hamilton_cycle_dp(threshold, deadline):
    """Use bitmask DP to check if a Hamilton cycle exists in the threshold subgraph.
    Returns (True, path) or (False, None)."""
    # Build adjacency as bitmasks
    adj_mask = [0] * n
    for i in range(threshold):
        u, v = edge_order[i]
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Check basic feasibility
    for v in range(n):
        if bin(adj_mask[v]).count('1') < 2:
            return False, None
    
    # DP: dp[mask][last] = True if there's a path from 0 visiting all vertices in 'mask' ending at 'last'
    # mask includes 0 and last
    # We use a list of dicts or a 2D structure. For n=24, this is too large for full DP.
    # Instead, use a smarter approach: DFS with strong pruning using bitmasks.
    
    # Use iterative deepening with bitmask-based pruning
    # Start from vertex 0
    full_mask = (1 << n) - 1
    
    # Precompute for quick degree checks
    # For each vertex, count how many neighbors it has in the current available set
    
    # Use a recursive DFS with bitmask state and aggressive pruning
    # State: (current_vertex, visited_mask)
    # But we need to track the path for reconstruction
    
    # Better approach: use a priority-queue based search or just optimized DFS
    # with memoization on (current, visited_mask) -> set of possible "next" vertices
    
    # For n=24, full DP is 2^24 * 24 = ~400M states, too much.
    # Use DFS with strong pruning instead.
    
    start = 0
    visited = 1 << start
    path = [start]
    
    # Memo: (current, visited) -> False if we've determined no solution from this state
    memo = set()
    
    def dfs(current, visited_mask, deadline):
        if time.time() > deadline:
            return False
        
        if visited_mask == full_mask:
            # Check if we can close the cycle
            if adj_mask[current] & (1 << start):
                return True
            return False
        
        key = (current, visited_mask)
        if key in memo:
            return False
        
        # Get available neighbors
        available = adj_mask[current] & ~visited_mask
        
        # Pruning: for each unvisited vertex, check it can still have degree >= 2
        # in the final cycle. The available "slots" for vertex w are:
        # - neighbors in unvisited set
        # - start (if w adjacent to start)
        # - current (if w adjacent to current)
        # - the vertex that will come after w in the path (unknown)
        # Simple check: unvisited neighbors + start + current >= 2
        
        unvisited = full_mask & ~visited_mask
        for w in range(n):
            if unvisited & (1 << w):
                # Count available connections for w
                avail_count = 0
                # Neighbors in unvisited
                nb_unvisited = adj_mask[w] & unvisited
                avail_count += bin(nb_unvisited).count('1')
                # Start vertex
                if adj_mask[w] & (1 << start):
                    avail_count += 1
                # Current vertex (the last in path)
                if adj_mask[w] & (1 << current):
                    avail_count += 1
                # Note: w also needs to connect to its successor in the path,
                # which is one of the unvisited vertices or start.
                # The above count includes start and current, plus unvisited neighbors.
                # But w's two cycle edges are: one to predecessor, one to successor.
                # Predecessor is either current or an earlier path vertex (but those are visited).
                # Actually, w's predecessor must be current or... no, w's predecessor
                # in the final cycle is the vertex just before w in the cycle.
                # This is getting complex. Let's use a simpler but effective check.
                if avail_count < 2:
                    memo.add(key)
                    return False
        
        # Order candidates: prefer vertices with fewer options (MRV heuristic)
        candidates = []
        temp_mask = available
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            v = lsb.bit_length() - 1
            candidates.append(v)
            temp_mask ^= lsb
        
        # Sort by number of unvisited neighbors (fewer first)
        candidates.sort(key=lambda v: bin(adj_mask[v] & ~visited_mask).count('1'))
        
        for v in candidates:
            if time.time() > deadline:
                return False
            
            path.append(v)
            new_visited = visited_mask | (1 << v)
            
            if dfs(v, new_visited, deadline):
                return True
            
            path.pop()
        
        memo.add(key)
        return False
    
    if dfs(start, visited, deadline):
        return True, path[:]
    return False, None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Binary search on threshold
    lo, hi = 1, num_edges
    best_cycle = None
    best_threshold = num_edges
    
    # First try to find a solution with binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        found, cycle = has_hamilton_cycle_dp(mid, deadline)
        if found:
            best_cycle = cycle
            best_threshold = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is None:
        # Fallback: try higher thresholds linearly from a reasonable starting point
        for t in range(max(1, best_threshold - 20), num_edges + 1):
            if time.time() > deadline:
                break
            found, cycle = has_hamilton_cycle_dp(t, deadline)
            if found:
                best_cycle = cycle
                best_threshold = t
                break
    
    if best_cycle is None:
        # Final fallback
        return list(range(n))
    
    return best_cycle
# EVOLVE-BLOCK-END
