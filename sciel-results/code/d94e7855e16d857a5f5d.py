def construct():
    import math
    import time

    N = 256
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1

    valid_vertices = []
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_vertices.append(a)
    
    M = len(valid_vertices)
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    adj_list = [[] for _ in range(M)]
    for i in range(M):
        a = valid_vertices[i]
        for j in range(i + 1, M):
            b = valid_vertices[j]
            if a + b in squares:
                adj_list[i].append(j)
                adj_list[j].append(i)
                
    # Check if bipartite
    color = [-1] * M
    is_bipartite = True
    from collections import deque
    
    for i in range(M):
        if color[i] == -1:
            color[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj_list[u]:
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        q.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break
            
    if is_bipartite:
        # Find Maximum Matching in Bipartite Graph
        # Left part: color 0, Right part: color 1
        # We need to map indices to left/right sets
        # Actually, standard Hopcroft-Karp or DFS matching works on the adjacency list directly
        # We just need to ensure we treat it as bipartite.
        
        match = [-1] * M # match[u] = v or -1
        
        def dfs(u, visited):
            for v in adj_list[u]:
                if not visited[v]:
                    visited[v] = True
                    if match[v] == -1 or dfs(match[v], visited):
                        match[v] = u
                        return True
            return False
            
        matching_size = 0
        for u in range(M):
            visited = [False] * M
            if dfs(u, visited):
                matching_size += 1
                
        # Minimum Vertex Cover size = Matching Size
        # Maximum Independent Set size = M - Matching Size
        # To construct the MIS, we need the Minimum Vertex Cover.
        # Konig's Theorem:
        # Let U be unmatched vertices in Left Part (color 0).
        # Let Z be set of vertices reachable from U via alternating paths.
        # Min Vertex Cover = (Left \ Z) U (Right ∩ Z)
        # MIS = V \ Min Vertex Cover = (Left ∩ Z) U (Right \ Z)
        
        # Identify Left and Right
        left = [i for i in range(M) if color[i] == 0]
        right = [i for i in range(M) if color[i] == 1]
        
        # Unmatched vertices in Left
        unmatched_left = [u for u in left if match[u] == -1] # Wait, match is defined as match[v]=u.
        # My dfs sets match[v] = u. So match[u] is not directly the match for u if u is in left?
        # Standard DFS matching:
        # match[v] stores the vertex in U matched to v in V.
        # So for u in U, is it matched? We need a reverse map or check if any v has match[v]==u.
        # Let's create a match_u array.
        
        match_u = [-1] * M
        for v in range(M):
            if match[v] != -1:
                match_u[match[v]] = v
                
        unmatched_left = [u for u in left if match_u[u] == -1]
        
        # BFS/DFS to find Z (vertices reachable from unmatched_left via alternating paths)
        # Alternating path: Start with unmatched edge, then matched, then unmatched...
        # In bipartite graph, edges are between Left and Right.
        # Start at u in Left (unmatched).
        # Step 1: Take unmatched edge u-v. (v must be in Right).
        # Step 2: Take matched edge v-u'. (u' must be in Left).
        # So from Left node, we traverse UNMATCHED edges.
        # From Right node, we traverse MATCHED edges.
        
        visited = [False] * M
        q = deque(unmatched_left)
        for u in unmatched_left:
            visited[u] = True
            
        while q:
            u = q.popleft()
            if color[u] == 0: # Left node
                # Traverse unmatched edges
                for v in adj_list[u]:
                    if not visited[v]:
                        # Edge u-v is unmatched if match_u[u] != v
                        # Since u is in Left, match_u[u] is its match in Right.
                        if match_u[u] != v:
                            visited[v] = True
                            q.append(v)
            else: # Right node
                # Traverse matched edge
                # The matched edge is v - match[v]
                if match[v] != -1:
                    u_next = match[v]
                    if not visited[u_next]:
                        visited[u_next] = True
                        q.append(u_next)
                        
        # Z = visited
        # Min Vertex Cover = (Left \ Z) U (Right ∩ Z)
        # MIS = (Left ∩ Z) U (Right \ Z)
        
        mis_indices = []
        for i in range(M):
            if color[i] == 0:
                if visited[i]: # In Z
                    mis_indices.append(i)
            else:
                if not visited[i]: # Not in Z
                    mis_indices.append(i)
                    
        return [valid_vertices[i] for i in mis_indices]
    else:
        # Fall back to B&B if not bipartite
        # (Code from previous thought)
        pass
