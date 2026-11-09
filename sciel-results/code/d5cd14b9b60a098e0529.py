"""
Maximum Clique Subdivision in a Pinned Graph.
Strategy: Find the maximum clique in the graph.
A clique of size k provides k branch vertices.
Paths between branch vertices are the direct edges (length 1).
Uses Bron-Kerbosch with pivot, degeneracy ordering, and Tomita coloring bound.
"""
import time
from itertools import combinations
from collections import deque

# The fixed problem instance
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degeneracy_order(adj):
    n = P['n']
    remaining = list(range(n))
    remaining_set = set(remaining)
    order = []
    while remaining:
        min_deg = min(len(adj[v] & remaining_set) for v in remaining)
        v = next(v for v in remaining if len(adj[v] & remaining_set) == min_deg)
        order.append(v)
        remaining.remove(v)
        remaining_set.discard(v)
    return order

def _color_sort(vertices, adj, deadline):
    """
    Greedy coloring to produce an upper bound and ordering for Tomita's algorithm.
    Returns a list of (color_index, vertex) sorted by color descending.
    """
    if not vertices:
        return []
    
    uncolored = list(vertices)
    # Sort by degree descending for better coloring
    uncolored.sort(key=lambda v: -len(adj[v]))
    
    color_classes = []
    
    while uncolored:
        if time.time() > deadline:
            break
        current_class = []
        i = 0
        while i < len(uncolored):
            v = uncolored[i]
            can_add = True
            for u in current_class:
                if u in adj[v]:
                    can_add = False
                    break
            if can_add:
                current_class.append(v)
                uncolored.pop(i)
            else:
                i += 1
        color_classes.append(current_class)
    
    result = []
    for ci, cls in enumerate(color_classes):
        for v in cls:
            result.append((ci, v))
    
    # Sort by color descending
    result.sort(key=lambda x: (-x[0], x[1]))
    return result

def _tomita_clique(adj, R, P_set, X_set, best_clique, deadline):
    """
    Tomita's maximum clique algorithm with coloring.
    R: current clique (set)
    P_set: candidate vertices (set)
    X_set: excluded vertices (set) - not strictly needed for max clique finding if we just want size, but good for structure.
    """
    if time.time() > deadline:
        return best_clique
    
    if not P_set:
        if len(R) > len(best_clique):
            best_clique = list(R)
        return best_clique
    
    if len(R) + len(P_set) <= len(best_clique):
        return best_clique
    
    # Get colored vertices
    colored = _color_sort(list(P_set), adj, deadline)
    
    # Process vertices in reverse order of coloring (highest color first)
    for color, v in reversed(colored):
        if time.time() > deadline:
            return best_clique
        
        # Upper bound check
        if len(R) + color + 1 <= len(best_clique):
            break
        
        new_R = R | {v}
        new_P_set = P_set & adj[v]
        
        best_clique = _tomita_clique(adj, new_R, new_P_set, X_set, best_clique, deadline)
        
        # Remove v from P_set and add to X_set
        P_set.discard(v)
        X_set.add(v)
    
    return best_clique

def _max_clique(adj):
    n = P['n']
    deadline = time.time() + 50  # Safety margin
    
    order = _degeneracy_order(adj)
    best = []
    
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
        
        # P = neighbors of v that come after v in degeneracy order
        P_set = set()
        for w in adj[v]:
            if w in order[i+1:]:
                P_set.add(w)
        
        if not P_set:
            continue
        
        # X = neighbors of v that come before v in degeneracy order
        X_set = set()
        for w in adj[v]:
            if w in order[:i]:
                X_set.add(w)
        
        best = _tomita_clique(adj, {v}, P_set, X_set, best, deadline)
    
    if not best:
        # Fallback if no edges or something weird
        best = _tomita_clique(adj, set(), set(range(n)), set(), best, deadline)
        
    return best

def _find_simple_path(adj, u, v, forbidden):
    """Find a simple path from u to v avoiding forbidden vertices (except endpoints)."""
    if u == v:
        return [u]
    
    visited = {u}
    queue = deque([(u, [u])])
    
    while queue:
        node, path = queue.popleft()
        if node == v:
            return path
        for nb in adj[node]:
            if nb not in visited and nb not in forbidden:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    
    return None

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    adj = _build_adj()
    clique = _max_clique(adj)
    
    # Ensure we have at least 2 branches
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    
    clique = sorted(clique)
    branches = list(clique)
    
    # Build paths: for each pair of branches, find a simple path
    # Since they form a clique, each pair is directly adjacent
    paths = []
    branch_set = set(branches)
    
    # Use combinations to ensure unordered pairs
    pairs = list(combinations(clique, 2))
    
    for u, v in pairs:
        if time.time() > deadline:
            # Fallback: just use direct edges for remaining
            for rem_u, rem_v in pairs:
                if [rem_u, rem_v] not in paths and [rem_v, rem_u] not in paths:
                    paths.append([rem_u, rem_v])
            break
        
        # Since it's a clique, direct edge should exist
        if v in adj[u]:
            paths.append([u, v])
        else:
            # Find path avoiding other branches
            forbidden = branch_set - {u, v}
            path = _find_simple_path(adj, u, v, forbidden)
            if path:
                paths.append(path)
            else:
                # Last resort: direct edge (shouldn't happen for clique)
                paths.append([u, v])
    
    # Verify count
    expected_count = len(pairs)
    if len(paths) != expected_count:
        # Rebuild paths with fallback to ensure validity
        paths = []
        for u, v in pairs:
            if v in adj[u]:
                paths.append([u, v])
            else:
                forbidden = branch_set - {u, v}
                path = _find_simple_path(adj, u, v, forbidden)
                if path:
                    paths.append(path)
                else:
                    paths.append([u, v])
    
    return {'branches': branches, 'paths': paths}
