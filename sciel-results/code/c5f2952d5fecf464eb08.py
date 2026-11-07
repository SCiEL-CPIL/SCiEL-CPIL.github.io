"""Rewritten program for triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = P['n']
    base_edges = P['edges']
    
    # Build base adjacency
    base_adj = [[] for _ in range(n)]
    base_edges_set = set()
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
    
    # Candidate edges: all (u,v) with u<v, not in base, and |u-v| >= 3 (triangle-free with base)
    # In a path, |u-v|==1 is base, |u-v|==2 shares a neighbor (triangle), |u-v|>=3 is fine.
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in base_edges_set or (v, u) in base_edges_set:
                continue
            if v - u >= 3:
                candidates.append((u, v))
    
    def _bfs_all(adj, n):
        """Return list of lists of distances from each node."""
        dist = [[-1]*n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[i][v] == -1:
                        dist[i][v] = dist[i][u] + 1
                        q.append(v)
        return dist
    
    def _diam(dist):
        return max(max(row) for row in dist)
    
    def _is_tri_free(adj, u, v):
        """Check if adding (u,v) keeps graph triangle-free. Assumes (u,v) not yet in adj."""
        nu = set(adj[u])
        nv = set(adj[v])
        return len(nu & nv) == 0
    
    def _verify(added_edges):
        adj = [list(base_adj[i]) for i in range(n)]
        seen = set()
        for u, v in added_edges:
            if (u,v) in base_edges_set or (v,u) in base_edges_set:
                return False
            if (u,v) in seen or (v,u) in seen:
                return False
            seen.add((u,v)); seen.add((v,u))
            adj[u].append(v); adj[v].append(u)
        # triangle-free
        for i in range(n):
            for j in adj[i]:
                for k in adj[j]:
                    if k != i and adj[i].count(k) > 0:
                        # Need to check if k is actually in adj[i] as a neighbor
                        # adj[i].count(k) > 0 means k is a neighbor of i
                        # But we also need (i,j) and (j,k) to be edges, which they are by iteration
                        # And (i,k) must NOT be an edge for no triangle... wait
                        # Triangle: i-j, j-k, k-i all edges.
                        # j is in adj[i], k is in adj[j]. If k is also in adj[i], then triangle.
                        # adj[i].count(k) > 0 checks if k in adj[i].
                        # But we should check set membership for efficiency.
                        pass
        # Proper triangle check
        for i in range(n):
            si = set(adj[i])
            for j in adj[i]:
                for k in adj[j]:
                    if k != i and k in si:
                        return False
        # Connected + diameter
        dist = _bfs_all(adj, n)
        for i in range(n):
            for j in range(n):
                if dist[i][j] == -1:
                    return False
        return _diam(dist) <= 4
    
    # Greedy construction to get initial upper bound
    def _greedy():
        adj = [list(base_adj[i]) for i in range(n)]
        added = []
        added_set = set()
        diam = 23  # path diameter
        while diam > 4:
            best_edge = None
            best_diam = diam + 1
            for u, v in candidates:
                if (u,v) in added_set or (v,u) in added_set:
                    continue
                if not _is_tri_free(adj, u, v):
                    continue
                # Try adding
                adj[u].append(v); adj[v].append(u)
                d = _diam(_bfs_all(adj, n))
                adj[u].remove(v); adj[v].remove(u)
                if d < best_diam:
                    best_diam = d
                    best_edge = (u, v)
            if best_edge is None:
                return None
            u, v = best_edge
            adj[u].append(v); adj[v].append(u)
            added.append([u, v])
            added_set.add((u,v)); added_set.add((v,u))
            diam = best_diam
        return added
    
    greedy_sol = _greedy()
    best_sol = greedy_sol if greedy_sol else None
    best_size = len(best_sol) if best_sol else float('inf')
    
    # Iterative deepening DFS to find optimal
    # We check k = 1, 2, ..., best_size - 1
    # If found, return it (it's optimal since we go in increasing order)
    
    def _dfs(adj, added, added_set, depth, max_depth, current_diam):
        nonlocal deadline, best_sol, best_size
        
        if time.time() > deadline:
            return False
        
        if len(added) > best_size:
            return False
        
        if current_diam <= 4:
            if len(added) < best_size:
                best_sol = [e[:] for e in added]
                best_size = len(added)
            return True
        
        if depth == max_depth:
            return False
        
        # Evaluate candidates
        useful = []
        for u, v in candidates:
            if (u,v) in added_set or (v,u) in added_set:
                continue
            if not _is_tri_free(adj, u, v):
                continue
            adj[u].append(v); adj[v].append(u)
            nd = _diam(_bfs_all(adj, n))
            adj[u].remove(v); adj[v].remove(u)
            if nd < current_diam:
                useful.append((u, v, nd))
        
        useful.sort(key=lambda x: x[2])
        
        # Limit branching factor for tractability
        top_k = 12
        for u, v, nd in useful[:top_k]:
            adj[u].append(v); adj[v].append(u)
            added.append([u, v])
            added_set.add((u,v)); added_set.add((v,u))
            
            if _dfs(adj, added, added_set, depth+1, max_depth, nd):
                return True
            
            adj[u].remove(v); adj[v].remove(u)
            added.pop()
            added_set.remove((u,v)); added_set.remove((v,u))
        
        return False
    
    for k in range(1, best_size):
        if time.time() > deadline:
            break
        
        adj = [list(base_adj[i]) for i in range(n)]
        added = []
        added_set = set()
        
        if _dfs(adj, added, added_set, 0, k, 23):
            if best_sol is not None and len(best_sol) <= k:
                break
    
    if best_sol is None:
        return []
    
    # Final verification
    if _verify(best_sol):
        return [[int(u), int(v)] for u, v in best_sol]
    return []

# EVOLVE-BLOCK-END
