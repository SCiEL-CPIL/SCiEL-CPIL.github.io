"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 55
    
    # Try increasing number of colors starting from 3
    for k in range(3, 15):
        if time.time() > deadline - 3:
            break
        result = _solve_with_k_colors(n, adj, k, deadline)
        if result is not None:
            return result
    
    # Fallback
    return list(range(n))


def _solve_with_k_colors(n, adj, k, deadline):
    """Try to color graph with k colors (0..k-1) satisfying:
    1. Adjacent vertices have different colors
    2. Subgraph induced by any two colors is a forest
    """
    color = [-1] * n
    
    # Precompute: for each vertex, its neighbors
    # Use incremental union-find with rollback for each color pair
    
    # For k colors, there are k*(k-1)//2 pairs
    # For each pair, maintain a union-find structure
    
    class UF:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size
            self.log = []  # stack of (rx, ry, old_rank_rx) for rollback
        
        def find(self, x):
            # No path compression to allow rollback
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx = self.find(x)
            ry = self.find(y)
            if rx == ry:
                return False  # cycle
            # Union by rank
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            # Record state for rollback: (root_that_changed, root_attached, old_rank_of_root_that_changed)
            # We attach ry to rx.
            self.log.append((rx, ry, self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def rollback(self):
            if not self.log:
                return
            rx, ry, old_rank_rx = self.log.pop()
            self.parent[ry] = ry
            self.rank[rx] = old_rank_rx
    
    # Create UF for each color pair
    ufs = {}
    for i in range(k):
        for j in range(i+1, k):
            ufs[(i, j)] = UF(n)
    
    # Track which edges have been added to which UF (for rollback)
    # When we assign color c to vertex v, we need to add edges (v, u) to UF[(c, color[u])]
    # for all neighbors u with color[u] != -1 and color[u] != c
    
    def backtrack(idx, order):
        if time.time() > deadline - 1:
            return None
        
        if idx == n:
            return list(color)
        
        v = order[idx]
        
        # Try colors in a smart order
        for c in range(k):
            # Check adjacency constraint
            ok = True
            for u in adj[v]:
                if color[u] == c:
                    ok = False
                    break
            if not ok:
                continue
            
            # Assign color c to v
            color[v] = c
            
            # Add edges to UFs and check for cycles
            ufs_to_rollback = []
            valid = True
            for u in adj[v]:
                cu = color[u]
                if cu == -1 or cu == c:
                    continue
                # Add edge (v, u) to UF for pair {c, cu}
                key = (min(c, cu), max(c, cu))
                uf = ufs[key]
                if uf.union(v, u):
                    ufs_to_rollback.append(key)
                else:
                    # Cycle detected - rollback what we did and try next color
                    valid = False
                    break
            
            if not valid:
                # Rollback
                for key in reversed(ufs_to_rollback):
                    ufs[key].rollback()
                color[v] = -1
                continue
            
            result = backtrack(idx + 1, order)
            if result is not None:
                return result
            
            # Rollback
            for key in reversed(ufs_to_rollback):
                ufs[key].rollback()
            color[v] = -1
        
        return None
    
    # Use a good vertex ordering: start with high-degree vertices
    # Vertex 49 has degree 4, connected to corners
    # Let's use a static ordering based on the grid structure
    # Row-major order might work well for grid-like structures
    order = list(range(n))
    
    return backtrack(0, order)

# EVOLVE-BLOCK-END
