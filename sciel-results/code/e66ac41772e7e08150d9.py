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

def _all_pairs_dist(n, adj):
    dist = [[-1] * n for _ in range(n)]
    for src in range(n):
        dist[src][src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(n):
                if adj[u][v] and dist[src][v] == -1:
                    dist[src][v] = dist[src][u] + 1
                    queue.append(v)
    return dist

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    n = P['n']
    base_adj = _build_base_adj(n, P['edges'])
    max_d = 4
    
    # Generate all valid candidate edges (triangle-free wrt base)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not base_adj[u][v] and _can_add(base_adj, u, v, n):
                candidates.append((u, v))
    
    # Greedy set-cover: repeatedly pick edge that fixes most remaining bad pairs
    selected = []
    selected_set = set()
    current_adj = [row[:] for row in base_adj]
    
    while time.time() < deadline - 5.0:
        # Compute current all-pairs distances
        dist = _all_pairs_dist(n, current_adj)
        
        # Find bad pairs (dist > max_d)
        bad_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i][j] > max_d:
                    bad_pairs.append((i, j))
        
        if not bad_pairs:
            break
        
        # Score each candidate: how many bad pairs does it fix?
        best_edge = None
        best_fixes = 0
        
        for edge in candidates:
            if edge in selected_set:
                continue
            u, v = edge
            if not _can_add(current_adj, u, v, n):
                continue
            
            # Temporarily add edge and count fixes
            current_adj[u][v] = True
            current_adj[v][u] = True
            dist2 = _all_pairs_dist(n, current_adj)
            
            fixes = 0
            for (i, j) in bad_pairs:
                if dist2[i][j] <= max_d:
                    fixes += 1
            
            # Remove edge
            current_adj[u][v] = False
            current_adj[v][u] = False
            
            if fixes > best_fixes:
                best_fixes = fixes
                best_edge = edge
        
        if best_edge is None or best_fixes == 0:
            break
        
        selected.append(best_edge)
        selected_set.add(best_edge)
        u, v = best_edge
        current_adj[u][v] = True
        current_adj[v][u] = True
    
    # Verify
    if selected and _is_valid(n, selected, base_adj):
        return [list(e) for e in selected]
    
    # Fallback
    seed = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    if _is_valid(n, seed, base_adj):
        return [list(e) for e in seed]
    
    return [list(e) for e in seed]

# EVOLVE-BLOCK-END
