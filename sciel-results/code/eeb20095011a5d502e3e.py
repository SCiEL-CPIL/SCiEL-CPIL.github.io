"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from functools import lru_cache

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
M = len(EDGE_ORDER)  # 276

# Build adjacency: for each vertex, list of (neighbor, arrival_index)
adj = [[] for _ in range(N)]
for idx, (u, v) in enumerate(EDGE_ORDER, 1):
    adj[u].append((v, idx))
    adj[v].append((u, idx))
# Sort each vertex's neighbors by arrival index
for v in range(N):
    adj[v].sort(key=lambda x: x[1])

# Precompute edge position lookup
edge_pos = {}
for idx, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[frozenset((u, v))] = idx

# Precompute bitmask adjacency for fast operations
# adj_mask[v] = bitmask of neighbors (all edges)
adj_mask = [0] * N
for v in range(N):
    for (nbr, pos) in adj[v]:
        adj_mask[v] |= (1 << nbr)

# For a given K, adj_mask_k[v] = bitmask of neighbors with arrival <= K
def build_adj_mask_k(K):
    masks = [0] * N
    for v in range(N):
        m = 0
        for (nbr, pos) in adj[v]:
            if pos <= K:
                m |= (1 << nbr)
            else:
                break
        masks[v] = m
    return masks

def cycle_score(cycle):
    """Compute the max arrival index of edges in the cycle."""
    max_idx = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        e = frozenset((u, v))
        pos = edge_pos[e]
        if pos > max_idx:
            max_idx = pos
    return max_idx

def has_hamilton_cycle(K, deadline=None):
    """Check if a Hamilton cycle exists using only edges with arrival index <= K."""
    # Build bitmask adjacency using only edges with position <= K
    adj_k = build_adj_mask_k(K)
    
    # Quick check: every vertex must have degree >= 2
    for v in range(N):
        if bin(adj_k[v]).count('1') < 2:
            return None
    
    # Check connectivity using bitmask BFS
    visited = 1  # start with vertex 0
    frontier = 1
    while frontier:
        # Expand frontier: find all neighbors of frontier nodes not yet visited
        neighbors = 0
        f = frontier
        while f:
            lsb = f & (-f)
            v = lsb.bit_length() - 1
            f ^= lsb
            neighbors |= adj_k[v]
        neighbors &= ~visited
        visited |= neighbors
        frontier = neighbors
    if bin(visited).count('1') != N:
        return None
    
    # Backtracking search for Hamilton cycle with strong pruning using bitmasks
    path = [0]
    used_mask = 1  # bitmask of used vertices, vertex 0 is used
    
    # Precompute neighbor lists sorted by arrival for candidate ordering
    nbr_lists = []
    for v in range(N):
        nbrs = []
        for (nbr, pos) in adj[v]:
            if pos <= K:
                nbrs.append((nbr, pos))
        nbr_lists.append(nbrs)
    
    node_count = 0
    deadline_check_interval = 5000
    
    def check_feasible(remaining_mask, path):
        """Check that all unvisited vertices can still be part of a Hamilton cycle.
        remaining_mask: bitmask of unvisited vertices
        path: current path list
        """
        if remaining_mask == 0:
            # All vertices used, check if last vertex connects to vertex 0
            last = path[-1]
            return (adj_k[last] & 1) != 0  # vertex 0 is bit 0
        
        cur = path[-1]
        # Available neighbors for remaining vertices: other remaining + cur + (0 if last step)
        is_last = (bin(remaining_mask).count('1') == 1)
        
        # For each remaining vertex, count available neighbors
        rem = remaining_mask
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            
            # Neighbors of v that are: in remaining, or is cur, or is 0 (if closing)
            avail = adj_k[v] & (remaining_mask | (1 << cur))
            if is_last:
                avail |= 1  # can connect to vertex 0 for closure
            elif (adj_k[v] & 1) and len(path) == N - 1:
                avail |= 1  # can connect to vertex 0 for closure
            
            if bin(avail).count('1') < 2:
                return False
        
        # Connectivity check on remaining subgraph
        if bin(remaining_mask).count('1') > 1:
            # BFS from first remaining vertex
            first_lsb = remaining_mask & (-remaining_mask)
            first_v = first_lsb.bit_length() - 1
            seen = first_lsb
            frontier = first_lsb
            while frontier:
                neighbors = 0
                f = frontier
                while f:
                    lsb2 = f & (-f)
                    v2 = lsb2.bit_length() - 1
                    f ^= lsb2
                    neighbors |= adj_k[v2]
                neighbors &= remaining_mask & ~seen
                seen |= neighbors
                frontier = neighbors
            if seen != remaining_mask:
                return False
        
        return True
    
    def backtrack():
        nonlocal node_count, used_mask
        node_count += 1
        if deadline is not None and node_count % deadline_check_interval == 0:
            if time.time() > deadline:
                return False
        
        if len(path) == N:
            last = path[-1]
            if (adj_k[last] & 1) != 0:  # connects to vertex 0
                return True
            return False
        
        cur = path[-1]
        remaining_mask = ~used_mask & ((1 << N) - 1)
        
        # Get candidate neighbors from cur that are unvisited
        candidates = []
        for (nbr, pos) in nbr_lists[cur]:
            if not (used_mask & (1 << nbr)):
                candidates.append((nbr, pos))
        
        # Sort by arrival index for better pruning
        candidates.sort(key=lambda x: x[1])
        
        for w, w_pos in candidates:
            # Pruning: w must have at least 1 unvisited neighbor (or connect to 0 at end)
            w_remaining = adj_k[w] & remaining_mask & ~(1 << cur)
            if len(path) < N - 1:
                if bin(w_remaining).count('1') < 1:
                    continue
            else:
                # Last vertex: must connect to 0
                if (adj_k[w] & 1) == 0:
                    continue
            
            # Try placing w
            used_mask |= (1 << w)
            path.append(w)
            
            new_remaining = ~used_mask & ((1 << N) - 1)
            
            if check_feasible(new_remaining, path):
                if backtrack():
                    return True
            
            path.pop()
            used_mask &= ~(1 << w)
        
        return False
    
    if backtrack():
        return path
    return None

def _find_best_cycle():
    """Binary search on K to find minimum threshold, then extract the cycle."""
    deadline = time.time() + 75  # Leave 15s safety margin for construct() overhead
    
    # Binary search for minimum K
    lo, hi = 24, M
    best_K = None
    best_cycle = None
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        cycle = has_hamilton_cycle(mid, deadline)
        if cycle is not None:
            best_K = mid
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: try higher K values directly
    for K in range(lo, min(lo + 30, M + 1)):
        if time.time() > deadline:
            break
        cycle = has_hamilton_cycle(K, deadline)
        if cycle is not None:
            return cycle
    
    # Final fallback: greedy construction
    path = [0]
    used = [False] * N
    used[0] = True
    
    while len(path) < N:
        cur = path[-1]
        best_nbr = None
        best_pos = float('inf')
        for (nbr, pos) in adj[cur]:
            if not used[nbr]:
                if pos < best_pos:
                    best_pos = pos
                    best_nbr = nbr
        if best_nbr is None:
            break
        path.append(best_nbr)
        used[best_nbr] = True
    
    if len(path) == N:
        last = path[-1]
        first = path[0]
        if frozenset((last, first)) in edge_pos:
            return path
    
    # Last resort: identity permutation
    return list(range(N))

# Cache the result
_cached_result = None

def construct():
    global _cached_result
    if _cached_result is not None:
        return _cached_result
    
    cycle = _find_best_cycle()
    
    _cached_result = cycle
    return cycle

# EVOLVE-BLOCK-END
