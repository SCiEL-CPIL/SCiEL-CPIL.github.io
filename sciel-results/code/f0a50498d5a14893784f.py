"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_base_adj(n, base_edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _bfs_diameter(n, adj):
    max_diam = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(n):
                if adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d == -1:
                return -1
            if d > max_diam:
                max_diam = d
    return max_diam

def _is_valid(n, added_edges, base_adj):
    adj = [row[:] for row in base_adj]
    for u, v in added_edges:
        if not isinstance(u, int) or not isinstance(v, int):
            return False
        if u == v:
            return False
        if adj[u][v]:
            return False
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        adj[u][v] = True
        adj[v][u] = True
    diam = _bfs_diameter(n, adj)
    return diam <= 4

def _can_add(adj, u, v, n):
    if adj[u][v]:
        return False
    for w in range(n):
        if w != u and w != v and adj[u][w] and adj[v][w]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    n = P['n']
    base_adj = _build_base_adj(n, P['edges'])
    
    seed = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    best = list(seed)
    
    # Precompute valid single edges
    valid_single = []
    for u in range(n):
        for v in range(u + 1, n):
            if not base_adj[u][v] and _can_add(base_adj, u, v, n):
                valid_single.append((u, v))
    
    # Try to reduce from 8 to 7: remove one, replace with another
    for i in range(len(best)):
        if time.time() > deadline:
            break
        current = [e for j, e in enumerate(best) if j != i]
        current_set = set(current)
        
        for cand in valid_single:
            if time.time() > deadline:
                break
            if cand in current_set:
                continue
            new_edges = current + [cand]
            if _is_valid(n, new_edges, base_adj):
                best = new_edges
                break
        
        if len(best) < 8:
            break
    
    # Try to reduce from 7 to 6
    if len(best) == 7:
        for i in range(len(best)):
            if time.time() > deadline:
                break
            current = [e for j, e in enumerate(best) if j != i]
            current_set = set(current)
            
            for cand in valid_single:
                if time.time() > deadline:
                    break
                if cand in current_set:
                    continue
                new_edges = current + [cand]
                if _is_valid(n, new_edges, base_adj):
                    best = new_edges
                    break
            
            if len(best) < 7:
                break
    
    # Try to reduce from 6 to 5
    if len(best) == 6:
        for i in range(len(best)):
            if time.time() > deadline:
                break
            current = [e for j, e in enumerate(best) if j != i]
            current_set = set(current)
            
            for cand in valid_single:
                if time.time() > deadline:
                    break
                if cand in current_set:
                    continue
                new_edges = current + [cand]
                if _is_valid(n, new_edges, base_adj):
                    best = new_edges
                    break
            
            if len(best) < 6:
                break
    
    return [list(e) for e in best]

# EVOLVE-BLOCK-END
