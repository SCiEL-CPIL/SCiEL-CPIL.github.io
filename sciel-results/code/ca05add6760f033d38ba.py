"""Find Hamiltonian cycle minimizing max edge arrival index via branch-and-bound."""
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Build adjacency: for each edge (u,v), record its arrival index (1-based)
# edge_index[(u,v)] = arrival index
edge_index = {}
adj = {i: [] for i in range(N)}  # adj[i] = list of (neighbor, arrival_index)
for idx, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_index[(u, v)] = idx
    edge_index[(v, u)] = idx
    adj[u].append((v, idx))
    adj[v].append((u, idx))

# Sort adjacency by arrival index for efficiency
for i in range(N):
    adj[i].sort(key=lambda x: x[1])

def has_hamiltonian_cycle(max_idx):
    """Check if there's a Hamiltonian cycle using only edges with arrival index <= max_idx.
    Returns the cycle as a list of vertices if found, else None.
    """
    # Build filtered adjacency
    filtered_adj = {i: [] for i in range(N)}
    for i in range(N):
        for (nb, idx) in adj[i]:
            if idx <= max_idx:
                filtered_adj[i].append(nb)
    
    # Check minimum degree
    for i in range(N):
        if len(filtered_adj[i]) < 2:
            return None
    
    # DFS-based Hamiltonian cycle search with pruning
    # Fix vertex 0 as start to break rotational symmetry
    start = 0
    visited = [False] * N
    path = [start]
    visited[start] = True
    
    def dfs(pos):
        if pos == N - 1:
            # Check if last vertex connects back to start
            last = path[-1]
            if start in filtered_adj[last]:
                return True
            return False
        
        current = path[-1]
        # Pruning: check if remaining unvisited vertices form a connected component
        # (simple check: each unvisited vertex must have at least one unvisited neighbor or be adjacent to current or to start if it's the last)
        
        for nb in filtered_adj[current]:
            if not visited[nb]:
                visited[nb] = True
                path.append(nb)
                if dfs(pos + 1):
                    return True
                path.pop()
                visited[nb] = False
        
        # Additional pruning: if current is not the last step, ensure that
        # every unvisited vertex (except possibly the one that will connect to start)
        # still has a path back. Simple degree check on unvisited subgraph.
        return False
    
    if dfs(1):
        return path[:]
    return None

def find_min_threshold():
    """Binary search for the minimum max_idx such that a Hamiltonian cycle exists."""
    lo = N  # minimum possible is N (cycle of N edges)
    hi = len(EDGE_ORDER)  # maximum possible
    
    best_cycle = None
    while lo <= hi:
        mid = (lo + hi) // 2
        cycle = has_hamiltonian_cycle(mid)
        if cycle is not None:
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_cycle, lo

# Precompute the answer
_cycle, _threshold = find_min_threshold()
if _cycle is None:
    _cycle = list(range(N))

def construct():
    return list(_cycle)
# EVOLVE-BLOCK-END
