<<<<<<< SEARCH
def _girth_at_least(n, edges, g):
    """
    Check if the girth of the graph is at least g.
    Returns True if no cycle of length < g exists.
    """
    if g <= 1:
        return True
        
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for cycles of length 3 to g-1
    # We can do BFS from each node
    # Optimization: if n is large, this might be slow.
    # But n is at most ~6000.
    
    for start in range(n):
        dist = [-1] * n
        prev = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    queue.append(v)
                elif v != prev[u] and dist[v] < dist[u]:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # Note: This detects cycles where the path from start to u and start to v are disjoint except at start?
                    # No, BFS tree ensures unique path from start.
                    # If v is a neighbor of u, and v was visited earlier, and v is not the parent of u,
                    # then we have a cycle.
                    # The length of the cycle is dist[u] + dist[v] + 1.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
=======
def _girth_at_least(n, edges, g):
    """
    Check if the girth of the graph is at least g.
    Returns True if no cycle of length < g exists.
    Uses BFS from each vertex to detect cycles.
    """
    if g <= 1:
        return True
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each start vertex, do BFS and check for back edges
    # that create cycles shorter than g
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g - 1:
                continue
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    # Back edge or cross edge
                    # Cycle length is dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True
>>>>>>> REPLACE
