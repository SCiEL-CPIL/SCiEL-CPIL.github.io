"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
TOTAL_EDGES = len(EDGE_ORDER)
START_TIME = time.time()
TIME_LIMIT = 85.0

def _elapsed():
    return time.time() - START_TIME

def _build_adjacency(k):
    """Build adjacency masks for edges 1..k (one-based)."""
    masks = [0] * N
    for i in range(k):
        u, v = EDGE_ORDER[i]
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def _check_degree(masks):
    """Check each vertex has degree >= 2."""
    for u in range(N):
        if masks[u].bit_count() < 2:
            return False
    return True

def _check_connectivity(masks):
    """Check graph is connected."""
    visited = 1
    stack = [0]
    while stack:
        u = stack.pop()
        neighbors = masks[u] & ~visited
        while neighbors:
            lsb = neighbors & -neighbors
            v = lsb.bit_length() - 1
            visited |= lsb
            stack.append(v)
            neighbors ^= lsb
    return visited == (1 << N) - 1

def _dfs_hamiltonian(masks, start, deadline):
    """
    Find Hamiltonian cycle starting at 'start' using bitmask adjacency.
    Uses Warnsdorff's rule for ordering.
    Returns path (list of N vertices) or None.
    """
    if _elapsed() > deadline:
        return None
    
    full_mask = (1 << N) - 1
    start_bit = 1 << start
    
    path = [start]
    visited = start_bit
    
    def dfs(u, visited, path):
        if _elapsed() > deadline:
            return None
        
        if len(path) == N:
            # Check closing edge
            if (masks[u] >> start) & 1:
                return path[:]
            return None
        
        # Get unvisited neighbors
        candidates = masks[u] & ~visited
        
        # If no candidates, dead end
        if candidates == 0:
            return None
        
        # Warnsdorff: sort by number of unvisited neighbors (fewest first)
        cand_list = []
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            # Count unvisited neighbors of v
            deg = (masks[v] & ~visited).bit_count()
            cand_list.append((deg, v))
            temp ^= lsb
        
        cand_list.sort()
        
        for _, v in cand_list:
            new_visited = visited | (1 << v)
            path.append(v)
            res = dfs(v, new_visited, path)
            if res is not None:
                return res
            path.pop()
        
        return None
    
    return dfs(start, visited, path)

def _find_cycle_for_k(k, deadline):
    """Try to find a Hamiltonian cycle using only edges 1..k."""
    if _elapsed() > deadline:
        return None
    
    masks = _build_adjacency(k)
    
    if not _check_degree(masks):
        return None
    if not _check_connectivity(masks):
        return None
    
    # Try different starting vertices
    for start in range(N):
        if _elapsed() > deadline:
            break
        path = _dfs_hamiltonian(masks, start, deadline)
        if path is not None:
            return path
    
    return None

def construct():
    deadline = START_TIME + TIME_LIMIT
    
    # Binary search for minimal k
    lo = N  # Need at least N edges
    hi = TOTAL_EDGES
    
    best_k = hi
    best_path = None
    
    # First check if hi works
    if _elapsed() < deadline:
        path = _find_cycle_for_k(hi, deadline)
        if path is not None:
            best_path = path
            best_k = hi
        else:
            # Shouldn't happen, but fallback
            return list(range(N))
    
    # Binary search
    while lo <= hi and _elapsed() < deadline:
        mid = (lo + hi) // 2
        path = _find_cycle_for_k(mid, deadline)
        if path is not None:
            best_path = path
            best_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_path is not None:
        return best_path
    
    # Fallback: identity permutation
    return list(range(N))

# EVOLVE-BLOCK-END
