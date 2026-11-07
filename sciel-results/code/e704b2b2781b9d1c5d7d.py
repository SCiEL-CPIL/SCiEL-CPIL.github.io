"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(edges):
    adj = {i: set() for i in range(P['n'])}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(adj):
    for u in adj:
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    return True

def _bfs_diameter(adj):
    n = P['n']
    max_dist = 0
    for src in range(n):
        dist = {src: 0}
        queue = [src]
        for node in queue:
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        if len(dist) < n:
            return n + 1  # disconnected
        max_dist = max(max_dist, max(dist.values()))
    return max_dist

def _check_solution(added_edges):
    base = P['edges']
    all_edges = base + added_edges
    adj = _build_adj(all_edges)
    if not _is_triangle_free(adj):
        return False
    if _bfs_diameter(adj) > 4:
        return False
    return True

def _candidate_edges():
    """Generate all valid candidate edges (not in base, triangle-free with base)."""
    base_set = set()
    for u, v in P['edges']:
        base_set.add((min(u,v), max(u,v)))
    
    adj_base = _build_adj(P['edges'])
    candidates = []
    for u in range(P['n']):
        for v in range(u+1, P['n']):
            if (u, v) in base_set:
                continue
            # Check if adding (u,v) creates a triangle with base edges
            # u and v must not share a common neighbor in base
            common = adj_base[u] & adj_base[v]
            if common:
                continue
            candidates.append((u, v))
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    candidates = _candidate_edges()
    n_cand = len(candidates)
    
    # Try increasing number of added edges
    for k in range(1, 15):
        if time.time() > deadline:
            break
        
        # Use a greedy + local search approach
        # First, try a simple greedy: pick edges that most reduce diameter
        best_sol = None
        
        # Greedy construction
        def greedy():
            current_added = []
            adj = _build_adj(P['edges'])
            while True:
                if time.time() > deadline:
                    return None
                if _bfs_diameter(adj) <= 4 and _is_triangle_free(adj):
                    return current_added
                # Find the pair with maximum distance
                dist = _bfs_diameter(adj)
                if dist <= 4:
                    return current_added
                
                # Find farthest pair
                max_d = 0
                far_pair = None
                for src in range(P['n']):
                    d = {src: 0}
                    q = [src]
                    for node in q:
                        for nb in adj[node]:
                            if nb not in d:
                                d[nb] = d[node] + 1
                                q.append(nb)
                    if len(d) < P['n']:
                        return None
                    for v in d:
                        if d[v] > max_d:
                            max_d = d[v]
                            far_pair = (src, v)
                
                if far_pair is None:
                    return None
                
                u, v = far_pair
                # Try adding edge (u,v) or nearby edges
                added = False
                # Try the direct edge first, then alternatives
                for cand in candidates:
                    cu, cv = cand
                    # Check if this edge is still valid (triangle-free with current graph)
                    common = adj[cu] & adj[cv]
                    if common:
                        continue
                    # Try adding it
                    new_adj = {i: set(adj[i]) for i in range(P['n'])}
                    new_adj[cu].add(cv)
                    new_adj[cv].add(cu)
                    if _is_triangle_free(new_adj) and _bfs_diameter(new_adj) <= 4:
                        current_added.append([cu, cv])
                        adj = new_adj
                        added = True
                        break
                
                if not added:
                    # Try all candidates
                    for cand in candidates:
                        cu, cv = cand
                        common = adj[cu] & adj[cv]
                        if common:
                            continue
                        new_adj = {i: set(adj[i]) for i in range(P['n'])}
                        new_adj[cu].add(cv)
                        new_adj[cv].add(cu)
                        if _is_triangle_free(new_adj):
                            current_added.append([cu, cv])
                            adj = new_adj
                            added = True
                            break
                    if not added:
                        return None
            return current_added
        
        sol = greedy()
        if sol is not None and len(sol) <= k:
            if _check_solution(sol):
                best_sol = sol
                break
        
        if best_sol is not None:
            break
    
    if best_sol is not None:
        return best_sol
    
    # Fallback: return a known valid solution
    # Let's construct one manually
    # For a path of 24 vertices, we can add edges to create a structure with diameter 4
    # Try: add edges (0,4), (0,8), (0,12), (0,16), (0,20) - but check triangle-free
    # Actually, let's just return a reasonable solution
    fallback = [[0, 4], [0, 8], [0, 12], [0, 16], [0, 20]]
    if _check_solution(fallback):
        return fallback
    
    # Last resort: return empty (will be invalid but avoids crash)
    return []

# EVOLVE-BLOCK-END
