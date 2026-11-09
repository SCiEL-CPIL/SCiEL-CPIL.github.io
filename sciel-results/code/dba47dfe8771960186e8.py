"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_graph(added_edges):
    """Build adjacency from base path + added edges."""
    n = 24
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_diameter(adj):
    """Compute diameter using BFS from all vertices."""
    n = len(adj)
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1  # disconnected
        max_dist = max(max_dist, max(dist))
    return max_dist

def _is_triangle_free(adj):
    """Check if graph is triangle-free."""
    n = len(adj)
    adj_set = [set(a) for a in adj]
    for u in range(n):
        for i, v in enumerate(adj[u]):
            for w in adj[u][i+1:]:
                if w in adj_set[v]:
                    return False
    return True

def _is_valid(added_edges):
    """Check if added edges produce a valid graph."""
    # Check for duplicate edges
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    for u, v in added_edges:
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    adj = _build_graph(added_edges)
    if not _is_triangle_free(adj):
        return False
    diameter = _bfs_diameter(adj)
    if diameter < 0 or diameter > 4:
        return False
    return True

def _check_bipartite_edges(added_edges):
    """Verify all added edges are between even and odd vertices (bipartite)."""
    for u, v in added_edges:
        if (u % 2) == (v % 2):
            return False
    return True

def _fast_validate(added_edges):
    """Fast validation assuming bipartite edges (no triangle check needed)."""
    # Check duplicates
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    for u, v in added_edges:
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Build adjacency and check diameter
    n = 24
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS diameter check
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
    return True

def _try_remove_edge(added_edges, idx):
    """Try removing edge at index idx and check if still valid."""
    if len(added_edges) <= 1:
        return None
    candidate = added_edges[:idx] + added_edges[idx+1:]
    if _fast_validate(candidate):
        return candidate
    return None

def _try_swap_edge(added_edges, idx, new_u, new_v):
    """Try swapping edge at index idx with new edge (new_u, new_v)."""
    key_new = (min(new_u, new_v), max(new_u, new_v))
    key_old = (min(added_edges[idx][0], added_edges[idx][1]), max(added_edges[idx][0], added_edges[idx][1]))
    if key_new == key_old:
        return None
    
    # Check if new edge is a base edge
    for u, v in P['edges']:
        if (min(u,v), max(u,v)) == key_new:
            return None
    
    # Check if new edge already exists in added_edges (excluding the one being replaced)
    for i, (u, v) in enumerate(added_edges):
        if i != idx:
            if (min(u,v), max(u,v)) == key_new:
                return None
    
    candidate = added_edges[:idx] + [[new_u, new_v]] + added_edges[idx+1:]
    if _fast_validate(candidate):
        return candidate
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # First, establish a known valid 10-edge solution as fallback
    base_solution = [
        [12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
        [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]
    ]
    if not _fast_validate(base_solution):
        return []
    
    best_solution = base_solution
    
    # Phase 1: Try to find a 9-edge solution with single hub
    odd_vertices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    even_vertices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    
    found_9 = False
    for h in even_vertices:
        if time.time() > deadline:
            break
        for exclude in odd_vertices:
            if time.time() > deadline:
                break
            candidate = [[h, o] for o in odd_vertices if o != exclude]
            if len(candidate) != 9:
                continue
            if _fast_validate(candidate):
                best_solution = candidate
                found_9 = True
                break
        if found_9:
            break
    
    # Phase 2: If no 9-edge solution with single hub, try two-hub configurations
    if not found_9:
        for h1 in even_vertices:
            if time.time() > deadline:
                break
            for h2 in even_vertices:
                if h2 <= h1:
                    continue
                if time.time() > deadline:
                    break
                # Try different splits of odd vertices between h1 and h2
                # h1 gets k odds, h2 gets (9-k) odds, total 9, no overlap
                for k in range(1, 9):
                    if time.time() > deadline:
                        break
                    # h1 connects to first k odd vertices, h2 to next (9-k)
                    if k + (9-k) > 12:
                        continue
                    candidate = [[h1, o] for o in odd_vertices[:k]] + \
                                 [[h2, o] for o in odd_vertices[k:k+9-k]]
                    if len(candidate) != 9:
                        continue
                    if _fast_validate(candidate):
                        best_solution = candidate
                        found_9 = True
                        break
                if found_9:
                    break
            if found_9:
                break
    
    # Phase 3: Local search - try to improve by removing edges from best solution
    if not found_9 and len(best_solution) > 9:
        # Try removing one edge at a time
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_solution)):
                if time.time() > deadline:
                    break
                candidate = _try_remove_edge(best_solution, i)
                if candidate is not None:
                    best_solution = candidate
                    improved = True
                    break
    
    # Phase 4: Try swapping edges to find better configurations
    if len(best_solution) == 10 and time.time() < deadline:
        # Try various swaps
        for i in range(len(best_solution)):
            if time.time() > deadline:
                break
            for new_u in even_vertices:
                if time.time() > deadline:
                    break
                for new_v in odd_vertices:
                    if time.time() > deadline:
                        break
                    candidate = _try_swap_edge(best_solution, i, new_u, new_v)
                    if candidate is not None:
                        best_solution = candidate
                        break
                if time.time() > deadline:
                    break
    
    # Ensure return type is correct
    result = []
    for e in best_solution:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
