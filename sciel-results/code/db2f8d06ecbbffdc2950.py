def _find_longest_cycle(adj, n):
    best_cycle = []
    vertices = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    num_starts = min(n, 5) # Reduced starts
    
    for start in vertices[:num_starts]:
        if not adj[start]:
            continue
        if time.time() > DEADLINE:
            break
            
        path = [start]
        on_path = {start}
        pos_in_path = {start: 0}
        
        def dfs(u):
            nonlocal best_cycle
            if time.time() > DEADLINE:
                return
                
            for v in sorted(adj[u], key=lambda x: -len(adj[x])):
                if v in on_path:
                    idx = pos_in_path[v]
                    cycle = path[idx:]
                    if len(cycle) > len(best_cycle):
                        best_cycle = cycle[:]
                else:
                    pos_in_path[v] = len(path)
                    path.append(v)
                    on_path.add(v)
                    
                    dfs(v)
                    
                    path.pop()
                    on_path.remove(v)
                    del pos_in_path[v]
                    
                    if len(best_cycle) >= n - 1:
                        return

        dfs(start)
        
        if len(best_cycle) >= n - 1:
            break
            
    return best_cycle if best_cycle else None
