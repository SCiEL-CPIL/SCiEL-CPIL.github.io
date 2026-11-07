# EVOLVE-BLOCK-START
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_distances(adj, src, n):
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _max_diameter(adj, n):
    maxd = 0
    for s in range(n):
        d = _bfs_distances(adj, s, n)
        if -1 in d: # Disconnected
            return float('inf')
        maxd = max(maxd, max(d))
    return maxd

def _is_triangle_free(adj, u, v, n):
    # Check if adding edge (u,v) creates a triangle
    # Triangle exists if there's a common neighbor of u and v
    nu = set(adj[u])
    for w in adj[v]:
        if w in nu:
            return False
    return True

def construct():
    n = P['n']
    base_edges = [tuple(e) for e in P['edges']]
    
    # Normalize base edges for fast lookup
    base_edge_set = set()
    for u, v in base_edges:
        base_edge_set.add((min(u,v), max(u,v)))
        
    adj = _build_adj(base_edges, n)
    
    # Generate all potential candidate edges
    # An edge is a candidate if it's not in base, and adding it keeps graph triangle-free
    # Note: We must check triangle-free against the *current* state during search,
    # but for the initial list, we can just list all non-base edges.
    # However, adding an edge that creates a triangle with *base* edges is always invalid.
    # Adding an edge that creates a triangle with *other added* edges is checked during search.
    
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edge_set:
                # Check if (u,v) creates a triangle with the BASE graph alone
                # If yes, it can never be added.
                if _is_triangle_free(adj, u, v, n):
                    candidates.append((u, v))
    
    # Sort candidates to help with symmetry breaking and potentially finding solutions faster
    # Sort by distance in base graph? Or just index.
    # Symmetry breaking: we pick edges in increasing order of their index in 'candidates'
    
    def dfs(depth, current_edges, current_adj, current_edge_set, start_idx):
        """
        depth: number of edges to add
        current_edges: list of added edges
        current_adj: adjacency list
        current_edge_set: set of normalized edges
        start_idx: minimum index in candidates to consider (for symmetry breaking)
        """
        if depth == 0:
            diam = _max_diameter(current_adj, n)
            return diam <= 4, current_edges
        
        for i in range(start_idx, len(candidates)):
            u, v = candidates[i]
            
            # Check if adding (u,v) to current graph creates a triangle
            # We need to check common neighbors in current_adj
            nu = set(current_adj[u])
            has_triangle = False
            for w in current_adj[v]:
                if w in nu:
                    has_triangle = True
                    break
            
            if has_triangle:
                continue
                
            # Add edge
            current_adj[u].append(v)
            current_adj[v].append(u)
            normalized = (min(u,v), max(u,v))
            current_edge_set.add(normalized)
            current_edges.append([u, v])
            
            # Recurse
            found, res = dfs(depth - 1, current_edges, current_adj, current_edge_set, i + 1)
            if found:
                return True, res
            
            # Undo
            current_edges.pop()
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            current_edge_set.discard(normalized)
            
        return False, []

    # Iterative Deepening
    max_depth = 6 # Upper bound, likely 3 or 4
    for d in range(1, max_depth + 1):
        # Deep copy initial state
        init_adj = [list(lst) for lst in adj]
        init_set = set(base_edge_set)
        found, res = dfs(d, [], init_adj, init_set, 0)
        if found:
            return res
            
    # Fallback: Greedy (should not happen if max_depth is sufficient)
    # But for safety, return an empty list or a known valid set if search fails
    # The problem guarantees a solution exists.
    return []
# EVOLVE-BLOCK-END
