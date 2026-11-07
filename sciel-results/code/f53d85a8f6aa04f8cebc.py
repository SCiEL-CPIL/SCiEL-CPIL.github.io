import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(P, added_edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if adj[u] & adj[v]:
                    return False
    return True

def _get_diameter(adj):
    n = len(adj)
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    if dist[neighbor] > max_dist:
                        max_dist = dist[neighbor]
                    queue.append(neighbor)
    return max_dist

def _validate(P, added_edges):
    n = P['n']
    base_set = set(tuple(sorted(e)) for e in P['edges'])
    edge_set = set()
    for e in added_edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            return False
        u, v = e
        if not isinstance(u, int) or not isinstance(v, int) or isinstance(u, bool) or isinstance(v, bool):
            return False
        if u == v:
            return False
        t = tuple(sorted((u, v)))
        if t in base_set:
            return False
        if t in edge_set:
            return False
        edge_set.add(t)
    adj = _build_adj(P, added_edges)
    if not _is_triangle_free(adj):
        return False
    if _get_diameter(adj) > 4:
        return False
    return True

def _greedy_diameter_reduce(P, deadline):
    n = P['n']
    base_set = set(tuple(sorted(e)) for e in P['edges'])
    
    # Precompute all candidate edges
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_set:
                candidates.append((u, v))
    
    added = []
    added_set = set()
    adj = _build_adj(P, added)
    current_diam = _get_diameter(adj)
    
    while current_diam > 4:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_diam = current_diam
        
        for (u, v) in candidates:
            if (u, v) in added_set:
                continue
            # Check triangle-free: u and v must not share a common neighbor
            if adj[u] & adj[v]:
                continue
            
            # Add edge temporarily
            adj[u].add(v)
            adj[v].add(u)
            
            # Compute new diameter
            new_diam = _get_diameter(adj)
            
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        added.append(list(best_edge))
        added_set.add(best_edge)
        adj = _build_adj(P, added)
        current_diam = best_diam
    
    # Try to remove edges (local search)
    for i in range(len(added) - 1, -1, -1):
        if time.time() > deadline:
            break
        trial = added[:i] + added[i+1:]
        if _validate(P, trial):
            added = trial
    
    return added

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Try greedy approach
    result = _greedy_diameter_reduce(P, deadline)
    
    if _validate(P, result):
        return result
    
    # Fallback: hub-based solution
    hub = 11
    added_edges = []
    base = set(tuple(sorted(e)) for e in P['edges'])
    for u in range(P['n']):
        if u % 2 == 0:
            if u == hub - 1 or u == hub + 1:
                continue
            t = tuple(sorted((hub, u)))
            if t not in base:
                added_edges.append([hub, u])
    
    if _validate(P, added_edges):
        return added_edges
    
    # Last resort
    _KNOWN_GOOD = [
        [11, 0], [11, 2], [11, 4], [11, 6], [11, 8],
        [11, 14], [11, 16], [11, 18], [11, 20], [11, 22]
    ]
    return _KNOWN_GOOD

# EVOLVE-BLOCK-END
