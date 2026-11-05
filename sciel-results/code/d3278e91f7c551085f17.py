from collections import deque

def _check_girth_bfs_sparse(adj, g):
    n = adj.shape[0]
    adj_csr = adj.tocsr()
    
    # Check all edges
    for u in range(n):
        start_idx = adj_csr.indptr[u]
        end_idx = adj_csr.indptr[u+1]
        for i in range(start_idx, end_idx):
            v = adj_csr.indices[i]
            if u >= v: continue
            
            # BFS from u to v, ignoring edge (u,v)
            visited = [-1] * n
            queue = deque()
            queue.append(u)
            visited[u] = 0
            
            found = False
            while queue:
                curr = queue.popleft()
                curr_dist = visited[curr]
                if curr_dist >= g - 1:
                    break
                
                # Neighbors of curr
                s_idx = adj_csr.indptr[curr]
                e_idx = adj_csr.indptr[curr+1]
                for j in range(s_idx, e_idx):
                    nb = adj_csr.indices[j]
                    # Skip the removed edge
                    if (curr == u and nb == v) or (curr == v and nb == u):
                        continue
                    
                    if visited[nb] == -1:
                        visited[nb] = curr_dist + 1
                        if nb == v:
                            found = True
                            break
                        queue.append(nb)
                if found:
                    break
            
            if found and visited[v] + 1 < g:
                return False
    return True
