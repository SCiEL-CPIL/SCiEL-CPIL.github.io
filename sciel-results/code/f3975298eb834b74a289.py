"""Find minimum bottleneck Hamiltonian cycle in a shuffled complete graph."""
import time
import random
import sys

# EVOLVE-BLOCK-START

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
M = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    edge_idx = [[-1] * N for _ in range(N)]
    for idx, (u, v) in enumerate(P['edge_order']):
        edge_idx[u][v] = idx
        edge_idx[v][u] = idx
    
    # Binary search on the bottleneck value B (1-based arrival index)
    # Check: does a Hamiltonian cycle exist using only edges with arrival <= B?
    # i.e., using only the first B edges in the edge_order list.
    
    # For a given B, build the subgraph with first B edges and check for Hamiltonian cycle.
    # Use backtracking with pruning.
    
    def has_hamiltonian_cycle(B):
        """Check if Hamiltonian cycle exists using only first B edges (1-based)."""
        # Build adjacency for subgraph
        adj = [[] for _ in range(N)]
        for i in range(B):
            u, v = P['edge_order'][i]
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degrees: each vertex must have degree >= 2
        for v in range(N):
            if len(adj[v]) < 2:
                return False
        
        # Backtracking: fix start at 0
        # State: path so far, visited set
        # Use bitmask for visited
        all_mask = (1 << N) - 1
        start = 0
        
        # Use DFS with pruning
        visited = 1 << start
        path = [start]
        
        def dfs(last, visited, path):
            if len(path) == N:
                # Check if we can close the cycle
                return start in adj[last]
            
            for w in adj[last]:
                if visited & (1 << w):
                    continue
                # Pruning: if w is not the last vertex to add, 
                # check that w can still reach the remaining vertices
                new_visited = visited | (1 << w)
                if new_visited == all_mask:
                    # This is the last vertex; check closing edge
                    if start in adj[w]:
                        return True
                else:
                    # Check that w has enough unvisited neighbors to continue
                    # (weak pruning)
                    path.append(w)
                    if dfs(w, new_visited, path):
                        return True
                    path.pop()
            return False
        
        return dfs(start, visited, path)
    
    # Binary search for minimum B
    lo, hi = 1, M
    best_B = M
    
    while lo <= hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        if has_hamiltonian_cycle(mid):
            best_B = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Now we know the minimum bottleneck is best_B.
    # We need to find an actual Hamiltonian cycle using only first best_B edges.
    
    def find_hamiltonian_cycle(B):
        """Find an actual Hamiltonian cycle using only first B edges."""
        adj = [[] for _ in range(N)]
        for i in range(B):
            u, v = P['edge_order'][i]
            adj[u].append(v)
            adj[v].append(u)
        
        all_mask = (1 << N) - 1
        start = 0
        visited = 1 << start
        path = [start]
        parent = {}  # (last, prev) -> None, just track path
        
        def dfs(last, visited, path):
            if len(path) == N:
                if start in adj[last]:
                    return True
                return False
            
            for w in adj[last]:
                if visited & (1 << w):
                    continue
                new_visited = visited | (1 << w)
                if new_visited == all_mask:
                    if start in adj[w]:
                        path.append(w)
                        return True
                else:
                    path.append(w)
                    if dfs(w, new_visited, path):
                        return True
                    path.pop()
            return False
        
        if dfs(start, visited, path):
            return path
        return None
    
    cycle = find_hamiltonian_cycle(best_B)
    if cycle is None:
        # Fallback: use identity permutation
        cycle = list(range(N))
    
    return cycle

# EVOLVE-BLOCK-END
