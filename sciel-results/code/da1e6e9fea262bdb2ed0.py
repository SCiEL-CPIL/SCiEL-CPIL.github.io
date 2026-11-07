import sys
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Precompute for each vertex, the list of (neighbor, edge_index)
# For fast bicolored cycle checking

def construct():
    deadline = time.time() + 85.0
    
    # Try backtracking for small K first
    for K in [3, 4, 5, 6, 7, 8]:
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        result = _backtrack(K, deadline)
        if result is not None:
            return [int(c) for c in result]
    
    # Fallback: should not reach here, but just in case
    return list(range(N))

def _backtrack(K, deadline):
    """
    Backtracking search with forward checking.
    Returns a valid coloring with K colors, or None.
    """
    colors = [-1] * N
    
    # Order: degree descending, then index for determinism
    order = sorted(range(N), key=lambda x: (-len(adj[x]), x))
    
    # For each color pair (a, b), maintain a DSU-like structure
    # Actually, we'll just do BFS/DFS check on the fly since N=50 is small
    # But we can be smarter: track components per color pair
    
    # For efficiency, we'll check bicolored cycles by:
    # When assigning color c to vertex v, for each neighbor u with color d != c,
    # check if there's already a path from v to u in the subgraph induced by {c, d}
    # (excluding v itself, since v is being added)
    
    # Actually simpler: after assigning v color c, check if the subgraph induced by
    # any pair (c, d) has a cycle. Since only edges incident to v are new, we just
    # need to check if any two neighbors of v with color d are already connected
    # in the {c,d} subgraph (without v).
    
    def has_bicolored_cycle(c, d):
        """Check if subgraph induced by colors c and d has a cycle."""
        # Collect vertices with color c or d
        verts = []
        for i in range(N):
            if colors[i] == c or colors[i] == d:
                verts.append(i)
        
        if len(verts) < 3:
            return False
        
        vset = set(verts)
        visited = set()
        
        def dfs(u, parent):
            visited.add(u)
            for w in adj[u]:
                if w not in vset:
                    continue
                if w == parent:
                    continue
                if w in visited:
                    return True  # cycle found
                if dfs(w, u):
                    return True
            return False
        
        for v in verts:
            if v not in visited:
                if dfs(v, -1):
                    return True
        return False
    
    def check_assignment(v, c):
        """Check if assigning color c to vertex v is valid (proper + no bicolored cycle)."""
        # Check proper coloring
        for u in adj[v]:
            if colors[u] == c:
                return False
        
        # Temporarily assign
        colors[v] = c
        
        # Check bicolored cycles for all pairs involving c
        for d in range(K):
            if d == c:
                continue
            if has_bicolored_cycle(c, d):
                colors[v] = -1
                return False
        
        colors[v] = -1  # Undo
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == N:
            return True
        
        v = order[idx]
        
        # Try colors in order 0, 1, ..., K-1
        for c in range(K):
            if check_assignment(v, c):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        
        return False
    
    if backtrack(0):
        return colors[:]
    return None
