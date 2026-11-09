<<<<<<< SEARCH
def hopcroft_karp(bipartite_adj):
    """
    Hopcroft-Karp algorithm for maximum bipartite matching.
    bipartite_adj: list of lists, where bipartite_adj[u] contains v for u in left, v in right.
    Returns: list of length len(left), where match_left[u] = v (or -1 if unmatched)
    """
    n_left = len(bipartite_adj)
    n_right = max([max(adj) for adj in bipartite_adj] + [0]) + 1
    
    match_left = [-1] * n_left
    match_right = [-1] * n_right
    dist = [0] * n_left
    
    def bfs():
        queue = deque()
        for u in range(n_left):
            if match_left[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
        
        found = False
        while queue:
            u = queue.popleft()
            for v in bipartite_adj[u]:
                if match_right[v] == -1:
                    found = True
                elif dist[match_right[v]] == float('inf'):
                    dist[match_right[v]] = dist[u] + 1
                    queue.append(match_right[v])
        return found

    def dfs(u):
        for v in bipartite_adj[u]:
            if match_right[v] == -1 or (dist[match_right[v]] == dist[u] + 1 and dfs(match_right[v])):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = float('inf')
        return False

    matching_size = 0
    while bfs():
        for u in range(n_left):
            if match_left[u] == -1:
                if dfs(u):
                    matching_size += 1
    return match_left
=======
def max_bipartite_matching(bipartite_adj):
    """
    Standard DFS-based maximum bipartite matching.
    bipartite_adj: list of lists, where bipartite_adj[u] contains v for u in left, v in right.
    Returns: list of length len(left), where match_left[u] = v (or -1 if unmatched)
    """
    n_left = len(bipartite_adj)
    if n_left == 0:
        return []
    
    # Determine n_right safely
    n_right = 0
    for adj in bipartite_adj:
        if adj:
            n_right = max(n_right, max(adj))
    n_right += 1
    
    match_left = [-1] * n_left
    match_right = [-1] * n_right
    
    def dfs(u, visited):
        for v in bipartite_adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_right[v] == -1 or dfs(match_right[v], visited):
                match_left[u] = v
                match_right[v] = u
                return True
        return False

    for u in range(n_left):
        visited = [False] * n_right
        dfs(u, visited)
        
    return match_left
>>>>>>> REPLACE

