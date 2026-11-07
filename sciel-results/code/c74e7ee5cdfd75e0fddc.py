    def solve(adj, remaining_candidates, current_edges, count):
        nonlocal best_solution, best_count
        
        if count >= best_count:
            return
            
        diam, fu, fv = get_diameter_and_pair(adj)
        if diam <= 4:
            if count < best_count:
                best_count = count
                best_solution = list(current_edges)
            return
            
        if not remaining_candidates:
            return
            
        def bfs_from(start):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
            return dist
            
        dist_fu = bfs_from(fu)
        dist_fv = bfs_from(fv)
        
        useful_edges = []
        for u, v in remaining_candidates:
            if creates_triangle(u, v, adj):
                continue
                
            d1 = dist_fu[u] + 1 + dist_fv[v]
            d2 = dist_fu[v] + 1 + dist_fv[u]
            
            if min(d1, d2) < diam:
                useful_edges.append((u, v))
                
        if not useful_edges:
            return
            
        # Optional: Sort useful_edges by how much they reduce diameter
        # This might help find a good solution faster
        useful_edges.sort(key=lambda e: min(dist_fu[e[0]] + 1 + dist_fv[e[1]], dist_fu[e[1]] + 1 + dist_fv[e[0]]))
        
        for u, v in useful_edges:
            adj[u].add(v)
            adj[v].add(u)
            current_edges.append((u, v))
            
            remaining_candidates.discard((u, v))
            solve(adj, remaining_candidates, current_edges, count + 1)
            remaining_candidates.add((u, v))
            
            current_edges.pop()
            adj[u].remove(v)
            adj[v].remove(u)
            
            if best_count == count + 1:
                # We found a solution with count+1 edges. 
                # Can we do better? Only if count+1 > 1. 
                # But we are minimizing. If we found 1, we stop.
                # If we found 2, we might find 1? No, because we are exploring depth-first.
                # Actually, if we found a solution of size K, we can prune any branch with count >= K.
                # The `if count >= best_count` at the start handles this.
                pass
