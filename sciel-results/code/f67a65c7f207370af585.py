def _check_new_cycle(n, adj, u, v, g):
    # Temporarily remove edge (u,v)
    adj[u].remove(v)
    adj[v].remove(u)
    
    # BFS from u to v with depth limit g-2
    dist = [-1] * n
    dist[u] = 0
    queue = deque([u])
    found = False
    while queue:
        x = queue.popleft()
        if dist[x] >= g - 2:
            break
        for y in adj[x]:
            if dist[y] == -1:
                dist[y] = dist[x] + 1
                if y == v:
                    found = True
                    break
                queue.append(y)
        if found:
            break
    
    # Restore edge (u,v)
    adj[u].append(v)
    adj[v].append(u)
    
    return found
