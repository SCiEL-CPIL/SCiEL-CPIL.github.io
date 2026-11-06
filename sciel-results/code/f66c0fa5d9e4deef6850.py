"""Find Hamilton cycle with minimal max edge arrival index."""
# EVOLVE-BLOCK-START
import time
import sys

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(u, v)] = i
    edge_pos[(v, u)] = i

def build_adj_masks(K):
    """Return list of bitmasks for each vertex using edges with position <= K."""
    masks = [0] * N
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        if i > K:
            break
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def popcount(x):
    return bin(x).count('1')

def find_hamilton_cycle(K, start=0, time_limit=30.0):
    """Find a Hamilton cycle using only edges with position <= K.
    Returns cycle as list of vertices, or None."""
    masks = build_adj_masks(K)
    
    # Quick check: every vertex must have degree >= 2
    for v in range(N):
        if popcount(masks[v]) < 2:
            return None
    
    full_mask = (1 << N) - 1
    start_bit = 1 << start
    
    # Precompute neighbor lists for faster iteration
    neighbors = []
    for v in range(N):
        nb_list = []
        for u in range(N):
            if masks[v] & (1 << u):
                nb_list.append(u)
        neighbors.append(nb_list)
    
    # Check that start connects to at least 2 vertices
    if len(neighbors[start]) < 2:
        return None
    
    deadline = time.time() + time_limit
    path = [start]
    visited = 1 << start
    current = start
    depth = 1
    
    # Use iterative DFS with Warnsdorff heuristic
    # State stack: (vertex, visited_mask, next_neighbor_index)
    stack = [(start, visited, 0)]
    
    while stack:
        if time.time() > deadline:
            return None
        
        v, vis, ni = stack[-1]
        nbs = neighbors[v]
        
        if ni >= len(nbs):
            # Backtrack
            stack.pop()
            if not stack:
                return None
            continue
        
        stack[-1] = (v, vis, ni + 1)
        nb = nbs[ni]
        
        if vis & (1 << nb):
            continue
        
        # If this is the last vertex, check if it connects back to start
        if depth == N - 1:
            if masks[nb] & (1 << start):
                # Found a cycle!
                result = path + [nb]
                return result
            continue
        
        # Pruning: after adding nb, check connectivity of unvisited vertices
        new_vis = vis | (1 << nb)
        
        # Check if nb has any unvisited neighbors (dead end check)
        if popcount(masks[nb] & ~new_vis) == 0 and depth < N - 2:
            continue
        
        # Connectivity check: ensure all unvisited vertices are in one component
        # with the path endpoints (v and nb)
        unvisited = full_mask & ~new_vis
        if unvisited:
            # BFS from nb through unvisited vertices
            # Actually, we need to check that unvisited vertices form a connected
            # subgraph that can be reached from both endpoints eventually.
            # Simpler check: each unvisited vertex must have at least one unvisited neighbor
            # or be adjacent to v or nb (the path endpoints)
            for u in range(N):
                if unvisited & (1 << u):
                    # u must have at least one unvisited neighbor, or be adjacent to v or nb
                    if popcount(masks[u] & unvisited) == 0:
                        # Check if u is adjacent to v or nb (can be inserted at end)
                        if not (masks[u] & (1 << v)) and not (masks[u] & (1 << nb)):
                            continue  # This is actually OK if depth is close to N-1
                        # For now, allow it but it's a weak check
                    # Stronger: if u has 0 unvisited neighbors AND is not adjacent to 
                    # either endpoint, it's stuck unless it's the last one to be placed
                    if popcount(masks[u] & unvisited) == 0 and depth < N - 3:
                        if not (masks[u] & (1 << v)) and not (masks[u] & (1 << nb)):
                            break
            else:
                # All unvisited vertices pass the basic check
                path.append(nb)
                stack.append((nb, new_vis, 0))
                depth += 1
                continue
            # If we broke out, this branch is pruned
            continue
        
        # If no unvisited vertices, we should have caught the last case above
        continue
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    # Binary search for minimum K
    lo, hi = N, len(EDGE_ORDER)
    best_cycle = None
    best_K = len(EDGE_ORDER) + 1
    
    # First, find a valid cycle at a reasonable K to have a fallback
    # Try K = 100, 150, 200, etc. to get a quick fallback
    for test_K in [100, 150, 200, 300, 400, 500, 600]:
        if time.time() > deadline - 10:
            break
        if test_K > len(EDGE_ORDER):
            continue
        cycle = find_hamilton_cycle(test_K, start=0, time_limit=5.0)
        if cycle is not None:
            best_cycle = cycle
            best_K = test_K
            break
    
    # Binary search between N and best_K (or len(EDGE_ORDER) if no fallback found)
    search_hi = min(best_K, len(EDGE_ORDER)) if best_cycle else len(EDGE_ORDER)
    lo, hi = N, search_hi
    
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        remaining = deadline - time.time()
        tl = min(20.0, max(2.0, remaining * 0.3))
        cycle = find_hamilton_cycle(mid, start=0, time_limit=tl)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Last resort fallbacks
    # Try a few more K values
    for K in range(100, len(EDGE_ORDER) + 1, 25):
        if time.time() > deadline - 2:
            break
        cycle = find_hamilton_cycle(K, start=0, time_limit=3.0)
        if cycle is not None:
            return cycle
    
    # Identity permutation as absolute last resort
    return list(range(N))
# EVOLVE-BLOCK-END
