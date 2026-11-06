"""Find the earliest Hamilton cycle in the pinned edge process on 24 vertices."""
import time
import sys

# The pinned edge order (from the seed program)
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

N = 24
TOTAL_EDGES = N * (N - 1) // 2  # 276

# Precompute: for each edge index (0-based), store the edge
# Build adjacency: for each vertex, list of (neighbor, edge_index)
_adj = [[] for _ in range(N)]
for i, (u, v) in enumerate(EDGE_ORDER):
    _adj[u].append((v, i))
    _adj[v].append((u, i))

# Sort adjacency by edge index for efficient prefix access
for v in range(N):
    _adj[v].sort(key=lambda x: x[1])


def _find_ham_cycle_in_prefix(k, deadline):
    """Find a Hamilton cycle using only the first k edges (indices 0..k-1).
    Returns the cycle as a list of vertices, or None."""
    
    # Build adjacency sets for the prefix
    # adj[v] = set of neighbors reachable within prefix k
    adj = [set() for _ in range(N)]
    for i in range(k):
        u, v = EDGE_ORDER[i]
        adj[u].add(v)
        adj[v].add(u)
    
    # Check minimum degree
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    # Use backtracking with vertex 0 fixed as start
    # We need a cycle of length N using exactly the edges in the prefix
    # Strategy: pick a starting vertex, do DFS with pruning
    
    # Order vertices by degree (ascending) for better pruning
    deg_order = sorted(range(N), key=lambda x: len(adj[x]))
    
    # Backtracking search for Hamilton cycle
    # Fix vertex 0 as the start to break symmetry
    start = 0
    
    # We'll build the path: path[0] = start, path[1], ..., path[N-1]
    # Then we need edge path[N-1] -> path[0] to close the cycle
    
    path = [0] * N
    used = [False] * N
    used[0] = True
    path[0] = 0
    
    # For efficiency, precompute for each vertex the sorted neighbor list
    # (by degree of neighbor, ascending) to guide search
    neighbor_order = [sorted(adj[v], key=lambda x: len(adj[x])) for v in range(N)]
    
    nodes_visited = 0
    max_nodes = 500000  # safety limit for a single backtracking attempt
    
    def backtrack(pos):
        nonlocal nodes_visited
        if time.time() > deadline:
            return None
        nodes_visited += 1
        if nodes_visited > max_nodes:
            return None
        
        if pos == N - 1:
            # Check if we can close the cycle
            last = path[N - 1]
            if start in adj[last]:
                return path[:]
            return None
        
        v = path[pos - 1]
        for nb in neighbor_order[v]:
            if not used[nb]:
                # Pruning: if nb is not the last vertex, it must have at least
                # one unused neighbor (other than v) to continue
                if pos < N - 2:
                    # Count unused neighbors of nb (excluding v)
                    unused_count = 0
                    for nn in adj[nb]:
                        if not used[nn] and nn != v:
                            unused_count += 1
                            if unused_count >= 1:
                                break
                    if unused_count == 0:
                        continue
                
                # Pruning: if nb is the last vertex, it must connect back to start
                if pos == N - 2:
                    if start not in adj[nb]:
                        continue
                
                used[nb] = True
                path[pos] = nb
                result = backtrack(pos + 1)
                if result is not None:
                    return result
                used[nb] = False
        
        return None
    
    result = backtrack(1)
    return result


def _score_cycle(cycle):
    """Compute the score (max 1-based edge index) for a given Hamilton cycle."""
    max_idx = 0
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        # Find edge index
        a, b = (u, v) if u < v else (v, u)
        # Need to find this edge in EDGE_ORDER
        # Precomputed lookup would be better
        idx = _edge_index.get((a, b))
        if idx is None:
            return float('inf')
        if idx + 1 > max_idx:
            max_idx = idx + 1
    return max_idx


# Precompute edge index lookup
_edge_index = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = (u, v) if u < v else (v, u)
    _edge_index[(a, b)] = i


def construct():
    """Find the Hamilton cycle with minimum score (earliest appearance)."""
    start_time = time.time()
    deadline = start_time + 80  # 80 second safety margin
    
    # Binary search on the threshold k
    # We want the smallest k such that a Hamilton cycle exists in the first k edges
    lo, hi = N, TOTAL_EDGES  # minimum edges needed for a cycle is N
    
    best_cycle = None
    best_score = float('inf')
    
    # First, do a quick check at the top to make sure we can find something
    # Start with binary search
    while lo < hi:
        if time.time() > deadline - 10:
            break
        mid = (lo + hi) // 2
        cycle = _find_ham_cycle_in_prefix(mid, deadline - 10)
        if cycle is not None:
            hi = mid
            if best_score > _score_cycle(cycle):
                best_score = _score_cycle(cycle)
                best_cycle = cycle
        else:
            lo = mid + 1
    
    # Now try to find a cycle at exactly lo (the minimum threshold)
    if time.time() < deadline - 5:
        cycle = _find_ham_cycle_in_prefix(lo, deadline - 5)
        if cycle is not None:
            s = _score_cycle(cycle)
            if s < best_score:
                best_score = s
                best_cycle = cycle
    
    # If we found a cycle, return it
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: try a simple approach - use the natural order
    # Check if identity permutation works
    identity = list(range(N))
    s = _score_cycle(identity)
    if s < float('inf'):
        return identity
    
    # Last resort: try to build a cycle greedily
    # Just return a valid permutation (even if score is high)
    return list(range(N))


# Pre-warm: verify the edge index lookup is complete
assert len(_edge_index) == TOTAL_EDGES
