import sys
import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add(tuple(sorted((u, v))))
    
    # Set of forbidden edges (tuples of sorted vertices)
    forbidden_edges = set()
    
    def find_cycle():
        """Find a Hamiltonian cycle avoiding forbidden_edges using DFS with pruning."""
        start = 0
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def dfs(u):
            if time.time() > deadline:
                return False
            
            if len(path) == n:
                # Check if u connects to start
                edge = tuple(sorted((u, start)))
                if edge not in forbidden_edges and edge in edge_set:
                    return True
                return False
            
            # Pruning: check connectivity of unvisited vertices
            unvisited = [v for v in range(n) if not visited[v]]
            if unvisited:
                # Check if unvisited vertices are connected (excluding start unless at end)
                # Simple check: can we reach all unvisited from u?
                if len(unvisited) > 0:
                    # BFS/DFS from u through unvisited vertices
                    reachable = set()
                    stack = [u]
                    while stack:
                        cur = stack.pop()
                        for w in adj[cur]:
                            if not visited[w] and w not in reachable:
                                reachable.add(w)
                                stack.append(w)
                    # All unvisited must be reachable from u
                    if len(reachable) < len(unvisited):
                        return False
            
            for v in adj[u]:
                if visited[v]:
                    continue
                edge = tuple(sorted((u, v)))
                if edge in forbidden_edges:
                    continue
                if edge not in edge_set:
                    continue
                
                # Pruning: if v would isolate some unvisited vertex, skip
                # Check if any unvisited vertex would have no path back
                # Simple heuristic: degree check
                # Count available neighbors for v (excluding visited and forbidden)
                avail_neighbors = 0
                for w in adj[v]:
                    if not visited[w] and w != u:
                        e = tuple(sorted((v, w)))
                        if e not in forbidden_edges and e in edge_set:
                            avail_neighbors += 1
                    elif w == u:
                        pass  # u is visited, this is the back edge
                    elif w == start and len(path) == n - 1:
                        # This is the final step
                        e = tuple(sorted((v, start)))
                        if e not in forbidden_edges and e in edge_set:
                            avail_neighbors += 1
                
                # If v has no available neighbors and we're not at the last step, skip
                if len(path) < n - 1 and avail_neighbors == 0:
                    continue
                
                visited[v] = True
                path.append(v)
                
                if dfs(v):
                    return True
                
                visited[v] = False
                path.pop()
            
            return False
        
        if dfs(start):
            return path[:]
        return None

    def try_greedy_cycle():
        """Try to find a cycle using greedy approach with backtracking."""
        start = 0
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def dfs(u, depth):
            if time.time() > deadline:
                return False
            
            if depth == n:
                edge = tuple(sorted((u, start)))
                if edge not in forbidden_edges and edge in edge_set:
                    return True
                return False
            
            # Sort neighbors by degree (fewest options first) for better pruning
            neighbors = []
            for v in adj[u]:
                if visited[v]:
                    continue
                edge = tuple(sorted((u, v)))
                if edge in forbidden_edges:
                    continue
                if edge not in edge_set:
                    continue
                # Count available options for v
                opts = 0
                for w in adj[v]:
                    if not visited[w] and w != u:
                        e = tuple(sorted((v, w)))
                        if e not in forbidden_edges and e in edge_set:
                            opts += 1
                neighbors.append((opts, v))
            
            neighbors.sort()
            
            for _, v in neighbors:
                visited[v] = True
                path.append(v)
                
                if dfs(v, depth + 1):
                    return True
                
                visited[v] = False
                path.pop()
            
            return False
        
        if dfs(start, 1):
            return path[:]
        return None

    cycles = []
    max_cycles = 3  # floor(70/21) = 3
    
    for i in range(max_cycles):
        if time.time() > deadline - 5:
            break
        
        cycle = find_cycle()
        if cycle is None:
            cycle = try_greedy_cycle()
        
        if cycle is None:
            break
        
        # Validate cycle
        valid = True
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e not in edge_set or e in forbidden_edges:
                valid = False
                break
        if not valid:
            continue
        
        cycles.append(cycle)
        
        # Mark edges in this cycle as forbidden
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            forbidden_edges.add(tuple(sorted((u, v))))
    
    # Fallback: ensure we return at least one valid cycle
    if not cycles:
        # The seed solution [0..20] is a valid Hamiltonian cycle
        # Verify: 0-1, 1-2, ..., 20-0 all exist in edge_set
        seed_cycle = list(range(n))
        valid = True
        for j in range(n):
            u = seed_cycle[j]
            v = seed_cycle[(j + 1) % n]
            e = tuple(sorted((u, v)))
            if e not in edge_set:
                valid = False
                break
        if valid:
            cycles.append(seed_cycle)
    
    # Ensure we return a non-empty list
    if not cycles:
        # Last resort: return a minimal valid cycle if one exists
        # Try to find any cycle of length >= 3
        for size in range(3, n + 1):
            found = False
            for start in range(n):
                for end in range(start + 1, n):
                    e = tuple(sorted((start, end)))
                    if e in edge_set:
                        # Try to find a path from start to end with size-2 intermediate vertices
                        # For simplicity, just return a triangle if one exists
                        pass
            
            # Look for triangles
            for a in range(n):
                for b in range(a + 1, n):
                    ab = tuple(sorted((a, b)))
                    if ab not in edge_set:
                        continue
                    for c in range(b + 1, n):
                        bc = tuple(sorted((b, c)))
                        ca = tuple(sorted((c, a)))
                        if bc in edge_set and ca in edge_set:
                            cycles.append([a, b, c])
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        
        if not cycles:
            # Absolute last resort: return empty list (will be invalid, but better than exception)
            return []
    
    return cycles
# EVOLVE-BLOCK-END
