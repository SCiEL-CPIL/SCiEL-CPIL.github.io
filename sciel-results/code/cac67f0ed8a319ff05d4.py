import time
import itertools

def construct():
    start_time = time.time()
    deadline = 80.0 # 80 seconds safety margin
    
    # Fallback: Two K3,3 components (Value 2)
    fallback_edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            fallback_edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            fallback_edges.append([u, v])
            
    best_value = 2
    best_edges = fallback_edges
    
    # Search for a better graph (Value 1)
    # We will try to find a 12-vertex 3-regular triangle-free graph
    # with only one distinct cycle length.
    
    n = 12
    # Adjacency list
    adj = [[] for _ in range(n)]
    edges = []
    
    def add_edge(u, v):
        adj[u].append(v)
        adj[v].append(u)
        edges.append([u, v])
        
    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        edges.remove([u, v])
        
    def has_triangle(u, v):
        # Check if u and v have a common neighbor
        common = set(adj[u]).intersection(set(adj[v]))
        return len(common) > 0
        
    def is_valid_partial():
        # Check degrees
        for i in range(n):
            if len(adj[i]) > 3:
                return False
        # Check triangles
        for i in range(n):
            for j in adj[i]:
                if has_triangle(i, j):
                    return False
        return True
        
    def get_cycle_lengths():
        # BFS/DFS to find cycles is expensive.
        # For 12 vertices, we can enumerate all simple cycles.
        # Use a standard algorithm for finding all cycles.
        # Since n is small, we can use a recursive DFS from each node.
        # To avoid duplicates, only start from the smallest node in the cycle.
        
        cycles = set()
        
        def dfs(node, start, path, visited):
            if len(path) > 1 and node == start:
                # Found a cycle
                # Normalize the cycle to avoid duplicates (rotation)
                # Only keep if start is the min node
                if start == min(path[:-1]):
                    cycles.add(tuple(sorted(path[:-1]))) # This is not unique enough
                    # Better: store the set of nodes? No, different cycles can have same nodes.
                    # Store the path sequence starting from min node.
                    pass
            
            # This simple DFS might be slow or miss cycles.
            # For 12 vertices, number of cycles is small.
            # We can just collect all simple paths and check for cycles.
            pass
            
        # Alternative: Use networkx? Not allowed.
        # Use a simple cycle enumeration.
        
        all_cycles = []
        
        def find_cycles(start):
            visited = [False] * n
            path = [start]
            visited[start] = True
            
            def backtrack(node):
                if len(path) > 1 and node == start:
                    # Found a cycle
                    # Check if start is the min node in the cycle to avoid duplicates
                    if start == min(path):
                        all_cycles.append(path[:])
                        return
                
                for neighbor in adj[node]:
                    if neighbor == start and len(path) > 2:
                        if start == min(path):
                            all_cycles.append(path + [start])
                        continue
                    if not visited[neighbor] and neighbor > start:
                        # Only visit nodes > start to ensure start is min
                        visited[neighbor] = True
                        path.append(neighbor)
                        backtrack(neighbor)
                        path.pop()
                        visited[neighbor] = False
                        
            backtrack(start)
            
        for i in range(n):
            find_cycles(i)
            
        if not all_cycles:
            return set()
            
        lengths = set()
        for c in all_cycles:
            lengths.add(len(c) - 1) # c includes start at end
            
        return lengths
        
    def search(depth, current_vertex):
        nonlocal best_value, best_edges
        
        if time.time() > deadline:
            return
            
        if depth == n * 3 // 2: # All degrees satisfied (3-regular on 12 verts has 18 edges)
            # Check if graph is valid (triangle-free)
            # It was checked during addition, but let's be sure.
            # Calculate cycle lengths
            lengths = get_cycle_lengths()
            if not lengths:
                return
            val = len(lengths)
            if val < best_value:
                best_value = val
                best_edges = [e[:] for e in edges]
                if val == 1:
                    return # Can't do better than 1
            return
            
        # Find next vertex to add an edge from
        # Heuristic: pick vertex with lowest degree
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= 3:
            return
            
        # Pick a vertex with min degree
        v = 0
        for i in range(n):
            if len(adj[i]) == min_deg:
                v = i
                break
                
        # Try to connect v to u
        for u in range(n):
            if u == v:
                continue
            if len(adj[u]) >= 3:
                continue
            if u in adj[v]:
                continue
            # Check triangle
            if has_triangle(v, u):
                continue
                
            # Add edge
            add_edge(v, u)
            
            search(depth + 1, u)
            
            # Remove edge
            remove_edge(v, u)
            
            if time.time() > deadline:
                return
                
    # Start search
    # To avoid symmetry, fix first few edges?
    # For now, just run. It might be slow.
    # Given 90s limit, and backtracking on 12 vertices might be too slow.
    # We will rely on the fallback.
    
    # However, the directive asks to TRY.
    # If the search is too slow, it will hit deadline and return fallback.
    
    search(0, 0)
    
    return best_edges
