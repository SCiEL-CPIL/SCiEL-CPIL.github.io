"""Find Hamilton cycle minimizing max edge arrival index in pinned edge order."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(10000)

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
NUM_EDGES = len(EDGE_ORDER)

# Precompute: for each edge (u,v), its 1-based position in the edge order
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    if u > v:
        u, v = v, u
    edge_pos[(u, v)] = i + 1

def get_edge_pos(u, v):
    if u > v:
        u, v = v, u
    return edge_pos.get((u, v), float('inf'))

def has_hamilton_cycle(threshold):
    """Check if a Hamilton cycle exists using only edges with position <= threshold.
    Returns the cycle as a list of vertices if found, else None."""
    
    # Build adjacency list for edges with position <= threshold
    adj = [[] for _ in range(N)]
    for (u, v), pos in edge_pos.items():
        if pos <= threshold:
            adj[u].append(v)
            adj[v].append(u)
    
    # Check minimum degree - every vertex must have degree >= 2
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # Use DFS to find Hamiltonian cycle starting from vertex 0
    # Fix vertex 0 as start to reduce symmetry
    start = 0
    visited = [False] * N
    path = [start]
    visited[start] = True
    
    def dfs(current):
        if len(path) == N:
            # Check if we can close the cycle back to start
            if start in adj[current]:
                return True
            return False
        
        for neighbor in adj[current]:
            if not visited[neighbor]:
                # Pruning: if this is the last vertex to visit, 
                # it must connect back to start
                if len(path) == N - 1:
                    if start not in adj[neighbor]:
                        continue
                
                visited[neighbor] = True
                path.append(neighbor)
                
                if dfs(neighbor):
                    return True
                
                path.pop()
                visited[neighbor] = False
        
        return False
    
    if dfs(start):
        return path
    return None

def construct():
    """Find Hamilton cycle minimizing the maximum edge arrival index."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Binary search on the threshold
    # Lower bound: at least 24 edges needed (one per vertex in cycle)
    # Upper bound: total number of edges
    lo = 1
    hi = NUM_EDGES
    
    best_cycle = None
    best_threshold = None
    
    # First, check if a Hamilton cycle exists at all with all edges
    # (it should, since K24 is complete)
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        cycle = has_hamilton_cycle(mid)
        
        if cycle is not None:
            best_cycle = cycle
            best_threshold = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is None:
        # Fallback: return a simple cycle using all edges
        # This shouldn't happen for a complete graph, but just in case
        return list(range(N))
    
    # Verify the cycle is valid and compute its actual score
    cycle = best_cycle
    # The cycle is [v0, v1, ..., v23], with implicit edge (v23, v0)
    max_pos = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        pos = get_edge_pos(u, v)
        if pos > max_pos:
            max_pos = pos
    
    # Return the cycle as a permutation of vertices 0..23
    # The evaluator expects a permutation describing the Hamilton cycle
    # with implicit closing edge
    return list(cycle)

# EVOLVE-BLOCK-END
