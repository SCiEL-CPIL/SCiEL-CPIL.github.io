def _has_girth_at_least(n, adj, g):
    max_depth = g // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = [start]
        qi = 0
        
        while qi < len(q):
            u = q[qi]
            qi += 1
            du = dist[u]
            if du >= max_depth:
                continue
                
            pu = parent[u]
            for v in adj[u]:
                if v == pu:
                    continue
                
                dv = dist[v]
                if dv == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                else:
                    if du + dv + 1 < g:
                        return False
                        
    return True
