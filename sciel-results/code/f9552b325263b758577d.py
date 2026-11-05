import time
import random
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s for final processing
    
    adj = [[] for _ in range(k + 1)] # Start with k+1 vertices? No.
    # Let's start with 1 vertex and add k neighbors.
    adj = [[]]
    degree = [0]
    
    for _ in range(k):
        new_v = len(adj)
        adj.append([])
        degree.append(0)
        adj[0].append(new_v)
        adj[new_v].append(0)
        degree[0] += 1
        degree[new_v] += 1
        
    active_vertices = [i for i in range(len(adj)) if degree[i] < k]
    
    while active_vertices:
        if time.time() > deadline:
            # Emergency completion
            while active_vertices:
                u = active_vertices.pop()
                if degree[u] >= k:
                    continue
                new_v = len(adj)
                adj.append([])
                degree.append(0)
                adj[u].append(new_v)
                adj[new_v].append(u)
                degree[u] += 1
                degree[new_v] += 1
                if degree[new_v] < k:
                    active_vertices.append(new_v)
            break
        
        u_idx = random.randint(0, len(active_vertices) - 1)
        u = active_vertices[u_idx]
        active_vertices[u_idx] = active_vertices[-1]
        active_vertices.pop()
        
        if degree[u] >= k:
            continue
            
        # BFS to find visited nodes
        visited = {u}
        queue = collections.deque([u])
        max_depth = g - 2
        current_depth = 0
        
        # Limit BFS time to 0.1s
        bfs_start = time.time()
        
        while queue and current_depth < max_depth:
            if time.time() - bfs_start > 0.1:
                break
            next_queue = []
            for node in queue:
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_queue.append(neighbor)
            queue = collections.deque(next_queue)
            current_depth += 1
            
        # Find valid v
        valid_v = None
        # To avoid scanning all N vertices, we can sample active_vertices
        # But for correctness, we should find one if it exists?
        # No, if we don't find one, we add a new vertex.
        # So we can just check a subset.
        
        # Let's check the first few active vertices
        # Shuffle to avoid bias?
        # Just pick a random subset of active vertices to check
        check_count = min(50, len(active_vertices))
        indices = random.sample(range(len(active_vertices)), check_count) if len(active_vertices) > check_count else range(len(active_vertices))
        
        for idx in indices:
            v = active_vertices[idx]
            if v == u:
                continue
            if v not in visited:
                valid_v = v
                break
                
        if valid_v is not None:
            adj[u].append(valid_v)
            adj[valid_v].append(u)
            degree[u] += 1
            degree[valid_v] += 1
        else:
            new_v = len(adj)
            adj.append([])
            degree.append(0)
            adj[u].append(new_v)
            adj[new_v].append(u)
            degree[u] += 1
            degree[new_v] += 1
            if degree[new_v] < k:
                active_vertices.append(new_v)
                
        if degree[u] < k:
            active_vertices.append(u)
            
    edges = []
    for i in range(len(adj)):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
                
    return len(adj), edges
