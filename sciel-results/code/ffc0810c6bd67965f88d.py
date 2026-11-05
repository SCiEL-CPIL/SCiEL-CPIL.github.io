def has_short_path(adj, u, v, limit):
    if u == v: return True # loop
    visited = {u: 0}
    dq = deque([(u, 0)])
    while dq:
        node, dist = dq.popleft()
        if dist >= limit:
            continue
        for nb in adj[node]:
            if nb == v:
                return True
            if nb not in visited:
                visited[nb] = dist + 1
                dq.append((nb, dist + 1))
    return False
