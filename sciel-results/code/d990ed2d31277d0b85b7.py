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

def find_min_bottleneck_hamilton_cycle(deadline):
    """
    Find the minimum bottleneck Hamiltonian cycle using Dijkstra-like search.
    State: (visited_mask, last_vertex)
    Cost: max edge position along the path
    We want to minimize the max edge position for a complete cycle.
    """
    import heapq
    
    full_mask = (1 << N) - 1
    
    # dp[mask][last] = minimum bottleneck cost to reach 'last' having visited 'mask'
    # Initialize: start at vertex 0 with only vertex 0 visited, cost 0
    # But we need a cycle, so we'll handle the closing edge at the end.
    
    # Use a priority queue: (cost, mask, last_vertex)
    # dp is a dict: (mask, last) -> min_cost
    dp = {}
    
    start = 0
    start_mask = 1 << start
    dp[(start_mask, start)] = 0
    
    heap = [(0, start_mask, start)]
    
    # Track parent for path reconstruction
    parent = {}  # (mask, last) -> (prev_mask, prev_last)
    
    while heap:
        if time.time() > deadline:
            break
            
        cost, mask, last = heapq.heappop(heap)
        
        # Check if this is a complete cycle
        if mask == full_mask:
            # Check if last connects back to start
            edge_key = (min(last, start), max(last, start))
            edge_cost = edge_pos.get(edge_key, 999999)
            total_cost = max(cost, edge_cost)
            
            # Reconstruct path
            path = [last]
            cur_mask, cur_last = mask, last
            while (cur_mask, cur_last) in parent:
                prev_mask, prev_last = parent[(cur_mask, cur_last)]
                path.append(prev_last)
                cur_mask, cur_last = prev_mask, prev_last
            path.reverse()
            # path[0] should be start
            return path, total_cost
        
        # Skip if we've found a better way to reach this state
        if (mask, last) in dp and dp[(mask, last)] < cost:
            continue
        
        # Try extending to unvisited vertices
        for nxt in range(N):
            if mask & (1 << nxt):
                continue  # already visited
            
            edge_key = (min(last, nxt), max(last, nxt))
            edge_cost = edge_pos.get(edge_key, 999999)
            if edge_cost == 999999:
                continue  # edge doesn't exist in the full graph (shouldn't happen for complete graph)
            
            new_cost = max(cost, edge_cost)
            new_mask = mask | (1 << nxt)
            new_state = (new_mask, nxt)
            
            if new_state not in dp or new_cost < dp[new_state]:
                dp[new_state] = new_cost
                parent[new_state] = (mask, last)
                heapq.heappush(heap, (new_cost, new_mask, nxt))
    
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin (evaluator kills at 90s)
    
    # Fallback: identity permutation
    best_cycle = list(range(N))
    
    # Use Dijkstra-like search to find minimum bottleneck Hamiltonian cycle
    cycle, cost = find_min_bottleneck_hamilton_cycle(deadline)
    
    if cycle is not None:
        # Verify it's a valid Hamilton cycle
        if len(set(cycle)) == N and len(cycle) == N:
            # Verify all edges exist
            valid = True
            for i in range(N):
                u = cycle[i]
                v = cycle[(i + 1) % N]
                key = (min(u, v), max(u, v))
                if key not in edge_pos:
                    valid = False
                    break
            if valid:
                return cycle
    
    return best_cycle
# EVOLVE-BLOCK-END
