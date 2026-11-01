import time
import random

def _random_k_regular(n, k):
    """Generate a random simple k-regular graph using configuration model."""
    if n * k % 2 != 0:
        return None
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    
    # Pair up stubs
    adjacency = [set() for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop, retry
        if v in adjacency[u]:
            return None # Multiple edge, retry
        adjacency[u].add(v)
        adjacency[v].add(u)
        
    # Verify degrees
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
            
    return adjacency

def _get_girth(adjacency):
    """Find the length of the shortest cycle in the graph."""
    n = len(adjacency)
    min_cycle = float('inf')
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        queue = [start]
        dist[start] = 0
        
        while queue:
            u = queue.pop(0)
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle == 3:
                        return 3
    return min_cycle if min_cycle != float('inf') else -1

def _find_short_cycle(adjacency, g):
    """Find a cycle of length < g. Returns list of nodes or None."""
    n = len(adjacency)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        queue = [start]
        dist[start] = 0
        
        while queue:
            u = queue.pop(0)
            if dist[u] >= g - 1:
                continue
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        path_u = []
                        curr = u
                        while curr != -1:
                            path_u.append(curr)
                            curr = parent[curr]
                        path_u_set = set(path_u)
                        
                        path_v = []
                        curr = v
                        while curr != -1:
                            path_v.append(curr)
                            curr = parent[curr]
                        
                        lca = -1
                        for node in path_u:
                            if node in path_v:
                                lca = node
                                break
                                
                        if lca == -1:
                            continue
                        
                        part1 = []
                        curr = u
                        while curr != lca:
                            part1.append(curr)
                            curr = parent[curr]
                        part1.append(lca)
                        
                        part2 = []
                        curr = v
                        while curr != lca:
                            part2.append(curr)
                            curr = parent[curr]
                        
                        cycle = part1 + list(reversed(part2))
                        
                        if len(cycle) < g:
                            return cycle
    return None

def _switching_construct(k, g, n, time_limit=10.0):
    start_time = time.time()
    
    # Try to generate a valid random graph
    while time.time() - start_time < 1.0:
        adj = _random_k_regular(n, k)
        if adj is not None:
            break
    else:
        return None
        
    current_girth = _get_girth(adj)
    
    # If already good, return
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
        
    # Main loop
    while time.time() - start_time < time_limit:
        # Find a short cycle
        cycle = _find_short_cycle(adj, g)
        if cycle is None:
            # No short cycle found? Girth >= g
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
        # Pick a random edge from the cycle
        m = len(cycle)
        if m < 3:
            return None
            
        edge_idx = random.randint(0, m - 1)
        u = cycle[edge_idx]
        v = cycle[(edge_idx + 1) % m]
        
        # Try to swap (u, v) with a random other edge (x, y)
        found_swap = False
        for _ in range(100):
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            
            if x == y or x == u or x == v or y == u or y == v:
                continue
                
            if y not in adj[x]:
                continue
                
            if x in adj[u]:
                continue
            if y in adj[v]:
                continue
            
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            new_girth = _get_girth(adj)
            
            if new_girth > current_girth:
                current_girth = new_girth
                found_swap = True
                break
            else:
                # Revert
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                
                adj[u].add(v)
                adj[v].add(u)
                adj[x].add(y)
                adj[y].add(x)
                
        if not found_swap:
            break
            
    return None

def construct(k, g):
    if k != 5 or g != 9:
        return None
        
    start_time = time.time()
    total_budget = 115.0
    
    target_ns = [1116, 1120, 1150, 1200, 1300, 1500, 2000]
    
    for n in target_ns:
        if time.time() - start_time > total_budget:
            break
            
        attempts = 0
        while time.time() - start_time < total_budget and attempts < 3:
            attempts += 1
            remaining = total_budget - (time.time() - start_time)
            if remaining < 5.0:
                break
            res = _switching_construct(k, g, n, time_limit=min(10.0, remaining))
            if res is not None:
                return res
                
    return None
