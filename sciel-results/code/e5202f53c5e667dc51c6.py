"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _bfs_all(adj, n):
    """BFS from all vertices, return max distance and whether connected."""
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
            return -1, False
        md = max(dist)
        if md > max_dist:
            max_dist = md
    return max_dist, True


def _is_valid(added_edges, base_edges, n):
    """Check all constraints: no dup, triangle-free (bipartite), connected, diameter<=4."""
    # Check no duplicate/existing edges
    existing = set()
    for u, v in base_edges:
        existing.add(frozenset((u, v)))
    for u, v in added_edges:
        fs = frozenset((u, v))
        if fs in existing:
            return False
        existing.add(fs)
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check bipartite (triangle-free)
    color = [-1] * n
    for start in range(n):
        if color[start] == -1:
            color[start] = 0
            stack = [start]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        stack.append(v)
                    elif color[v] == color[u]:
                        return False
    
    # Check diameter and connectivity
    max_d, connected = _bfs_all(adj, n)
    if not connected or max_d > 4:
        return False
    
    return True


def construct():
    deadline = time.time() + 80  # Safety margin
    
    n = P['n']
    base_edges = P['edges']
    
    # Base edge set for quick lookup
    base_set = set(frozenset((u, v)) for u, v in base_edges)
    
    # All candidate edges: opposite parity, not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if u % 2 != v % 2 and frozenset((u, v)) not in base_set:
                candidates.append((u, v))
    
    # Precompute: for each candidate edge, which vertex pairs does it help connect?
    # We'll use a greedy approach: start empty, add edges that most reduce max distance
    
    def _get_max_dist(adj, n):
        max_d = 0
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
                return 999
            md = max(dist)
            if md > max_d:
                max_d = md
        return max_d
    
    def _build_adj(added):
        adj = [[] for _ in range(n)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    # Greedy: repeatedly add the edge that most reduces the maximum distance
    added = []
    added_set = set()
    
    while time.time() < deadline - 20:
        adj = _build_adj(added)
        max_d = _get_max_dist(adj, n)
        if max_d <= 4:
            break
        
        # Find the edge that most reduces max distance
        best_edge = None
        best_new_max = max_d
        for edge in candidates:
            if edge in added_set:
                continue
            test_added = added + [edge]
            test_adj = _build_adj(test_added)
            test_max = _get_max_dist(test_adj, n)
            if test_max < best_new_max:
                best_new_max = test_max
                best_edge = edge
        
        if best_edge is None:
            break
        
        added.append(best_edge)
        added_set.add(best_edge)
    
    # Verify
    if not _is_valid(added, base_edges, n):
        # Fallback: try a structured solution
        # Connect each even vertex to specific odd vertices
        added = []
        for i in range(12):
            even_v = 2 * i
            # Connect to odd vertices at distance 1, 2, 3 in the path
            for offset in [1, 2, 3]:
                if i + offset < 12:
                    odd_v = 2 * (i + offset) + 1
                    if frozenset((even_v, odd_v)) not in base_set:
                        added.append([even_v, odd_v])
                if i - offset >= 0:
                    odd_v = 2 * (i - offset) + 1
                    if frozenset((even_v, odd_v)) not in base_set:
                        added.append([even_v, odd_v])
        # Remove duplicates
        seen = set()
        unique_added = []
        for e in added:
            fs = frozenset(e)
            if fs not in seen:
                seen.add(fs)
                unique_added.append(e)
        added = unique_added
    
    if not _is_valid(added, base_edges, n):
        # Last resort: complete bipartite minus base
        added = [list(e) for e in candidates]
    
    # Local search: try removing edges
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(added)):
            if time.time() > deadline - 10:
                break
            candidate = added[:i] + added[i+1:]
            if _is_valid(candidate, base_edges, n):
                added = candidate
                improved = True
                break
    
    # Try edge swaps: remove one, add another
    if time.time() < deadline - 5:
        for i in range(len(added)):
            if time.time() > deadline - 5:
                break
            for edge in candidates:
                if edge in added:
                    continue
                candidate = added[:i] + added[i+1:] + [list(edge)]
                if _is_valid(candidate, base_edges, n):
                    # Check if we can remove another edge now
                    for j in range(len(candidate)):
                        test = candidate[:j] + candidate[j+1:]
                        if _is_valid(test, base_edges, n):
                            added = test
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
    
    # Final removal pass
    improved2 = True
    while improved2 and time.time() < deadline - 3:
        improved2 = False
        for i in range(len(added)):
            candidate = added[:i] + added[i+1:]
            if _is_valid(candidate, base_edges, n):
                added = candidate
                improved2 = True
                break
    
    # Ensure proper types
    result = []
    for e in added:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
