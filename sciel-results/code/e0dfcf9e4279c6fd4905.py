import time

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edge_order = P['edge_order']
    total_edges = len(edge_order)
    
    # Build position map
    pos_map = {}
    for i, (u, v) in enumerate(edge_order):
        key = (min(u, v), max(u, v))
        pos_map[key] = i  # 0-based position
    
    def get_score(cycle):
        """Compute the score (max 1-based position) of a Hamilton cycle."""
        max_pos = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            key = (min(u, v), max(u, v))
            pos = pos_map[key] + 1  # 1-based
            if pos > max_pos:
                max_pos = pos
        return max_pos
    
    def has_hamilton_cycle(T, time_limit=5.0):
        """Check if there's a Hamilton cycle using only edges with position <= T (0-based)."""
        start_time = time.time()
        
        # Build adjacency list using only first T+1 edges
        adj = [set() for _ in range(n)]
        for i in range(T + 1):
            u, v = edge_order[i]
            adj[u].add(v)
            adj[v].add(u)
        
        # Check if any vertex has degree < 2
        for i in range(n):
            if len(adj[i]) < 2:
                return None
        
        # Backtracking to find Hamilton cycle starting from vertex 0
        visited = [False] * n
        visited[0] = True
        path = [0]
        
        def backtrack():
            if time.time() - start_time > time_limit:
                return None
            
            current = path[-1]
            if len(path) == n:
                # Check if we can close the cycle
                if 0 in adj[current]:
                    return path[:]
                return None
            
            # Try neighbors in order of their position (earlier first for better pruning)
            candidates = [v for v in adj[current] if not visited[v]]
            
            # Sort by degree of unvisited neighbors (least first - more constrained first)
            def unvisited_degree(v):
                count = 0
                for w in adj[v]:
                    if not visited[w]:
                        count += 1
                return count
            
            candidates.sort(key=unvisited_degree)
            
            for v in candidates:
                # Pruning: if v is not 0 and v's only unvisited neighbor is current, 
                # and there are still other unvisited vertices, this would strand others
                # More careful pruning:
                unvisited_neighbors = [w for w in adj[v] if not visited[w] and w != current]
                
                # If v is the last vertex to visit (all others visited after), 
                # it must connect back to 0
                if len(path) == n - 1:
                    if 0 not in adj[v]:
                        continue
                
                visited[v] = True
                path.append(v)
                
                result = backtrack()
                if result is not None:
                    return result
                
                visited[v] = False
                path.pop()
            
            return None
        
        result = backtrack()
        return result
    
    # Binary search for minimum T
    # Lower bound: at least 23 (need 24 edges, 0-indexed so positions 0-23)
    # Upper bound: total_edges - 1
    lo = 23
    hi = total_edges - 1
    best_cycle = None
    best_score = total_edges + 1
    
    # First, let's try to find a cycle with a reasonable upper bound
    # Start with a greedy approach to get an initial solution
    
    # Greedy: build cycle using earliest available edges
    def greedy_build():
        adj = [set() for _ in range(n)]
        for i, (u, v) in enumerate(edge_order):
            adj[u].add(v)
            adj[v].add(u)
        
        # Start from vertex 0, greedily pick earliest edge to unvisited vertex
        visited = [False] * n
        visited[0] = True
        path = [0]
        
        while len(path) < n:
            current = path[-1]
            # Find earliest unvisited neighbor
            best_v = None
            best_pos = float('inf')
            for v in adj[current]:
                if not visited[v]:
                    pos = pos_map[(min(current, v), max(current, v))]
                    if pos < best_pos:
                        best_pos = pos
                        best_v = v
            
            if best_v is None:
                return None
            
            visited[best_v] = True
            path.append(best_v)
        
        # Check if we can close the cycle
        if 0 in adj[path[-1]]:
            return path
        return None
    
    # Try greedy first
    greedy_cycle = greedy_build()
    if greedy_cycle is not None:
        s = get_score(greedy_cycle)
        if s < best_score:
            best_score = s
            best_cycle = greedy_cycle
    
    # Binary search
    for T in range(lo, hi + 1):
        if time.time() > 80:  # Safety margin
            break
        
        cycle = has_hamilton_cycle(T, time_limit=3.0)
        if cycle is not None:
            s = get_score(cycle)
            if s < best_score:
                best_score = s
                best_cycle = cycle
            # This T works, try smaller
            hi = T - 1
        else:
            lo = T + 1
        
        if lo > hi:
            break
    
    # Also try some specific values around the found range for better cycles
    if best_cycle is not None:
        # Try to improve by checking nearby T values
        for T in range(max(23, best_score - 5), min(total_edges - 1, best_score + 5)):
            if time.time() > 85:
                break
            cycle = has_hamilton_cycle(T, time_limit=2.0)
            if cycle is not None:
                s = get_score(cycle)
                if s < best_score:
                    best_score = s
                    best_cycle = cycle
    
    if best_cycle is None:
        # Fallback: return identity permutation
        return list(range(n))
    
    return best_cycle
