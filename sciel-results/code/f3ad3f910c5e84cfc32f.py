"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(edges):
    adj = [[] for _ in range(P['n'])]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_max_dist(adj):
    """Return max shortest-path distance (diameter) of the graph."""
    n = P['n']
    max_d = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    max_d = max(max_d, dist[v])
                    q.append(v)
    return max_d

def _is_triangle_free(edge_set, adj):
    """Check if adding edges in edge_set to base creates any triangle."""
    for u, v in edge_set:
        for w in adj[u]:
            if w != v and (w, v) in edge_set or (v, w) in edge_set:
                return False
    return True

def construct():
    deadline = time.time() + 80  # 80s budget, 10s safety margin
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e), max(e)))
    
    # All possible edges that are not in base
    all_possible = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_possible.append((u, v))
    
    # Precompute adjacency for base
    base_adj = _build_adj(P['edges'])
    
    # Try small numbers of added edges
    best_added = None
    best_count = len(all_possible) + 1  # infinity
    
    # First, try a greedy heuristic to get a good initial solution
    # Strategy: use 2 hubs, one even and one odd
    # Try all pairs of (even_hub, odd_hub)
    even_hubs = list(range(0, n, 2))  # 0,2,...,22
    odd_hubs = list(range(1, n, 2))   # 1,3,...,23
    
    def _try_solution(added_edges):
        """Check if adding these edges gives valid graph with diameter <= 4."""
        if not added_edges:
            return False
        # Build full edge set
        full_edges = list(P['edges']) + added_edges
        # Check for duplicates
        seen = set()
        for u, v in full_edges:
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        # Check triangle-free
        adj = _build_adj(full_edges)
        for u in range(n):
            for i, v in enumerate(adj[u]):
                for w in adj[u][i+1:]:
                    if w in adj[v]:
                        return False
        # Check diameter
        diam = _bfs_max_dist(adj)
        return diam <= 4
    
    # Try 2-hub strategies
    for eh in even_hubs:
        if time.time() > deadline:
            break
        for oh in odd_hubs:
            if time.time() > deadline:
                break
            # Add edges from eh to all odd vertices (except path neighbors)
            # and from oh to all even vertices (except path neighbors)
            added = []
            for v in odd_hubs:
                if v != eh and (min(eh,v), max(eh,v)) not in base_edges:
                    added.append((eh, v))
            for v in even_hubs:
                if v != oh and (min(oh,v), max(oh,v)) not in base_edges:
                    added.append((oh, v))
            # Remove duplicates
            seen = set()
            unique_added = []
            for u, v in added:
                key = (min(u,v), max(u,v))
                if key not in seen:
                    seen.add(key)
                    unique_added.append((u, v))
            if len(unique_added) < best_count and _try_solution(unique_added):
                best_added = unique_added
                best_count = len(unique_added)
    
    # If no 2-hub solution found, try 1-hub
    if best_added is None:
        for h in range(n):
            if time.time() > deadline:
                break
            added = []
            for v in range(n):
                if v != h and (min(h,v), max(h,v)) not in base_edges:
                    added.append((h, v))
            if len(added) < best_count and _try_solution(added):
                best_added = added
                best_count = len(added)
    
    # If still nothing, fall back to current known solution
    if best_added is None:
        added = []
        for u in range(0, n, 2):
            v = 11
            if u != v and (min(u,v), max(u,v)) not in base_edges:
                added.append((u, v))
        best_added = added
    
    return [[int(u), int(v)] for u, v in best_added]
# EVOLVE-BLOCK-END
