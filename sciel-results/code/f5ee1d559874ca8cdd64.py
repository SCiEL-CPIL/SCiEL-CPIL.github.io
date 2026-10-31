def solve_greedy(N, k, g, time_limit=5):
    start_time = time.time()
    
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
            
        deg = [0] * N
        adj = [[] for _ in range(N)]
        edge_set = set()
        edges = []
        
        # List of vertices with deg < k
        avail = list(range(N))
        
        success = True
        for step in range(N * k // 2):
            if not avail:
                break
                
            u = random.choice(avail)
            
            found_v = False
            # Try to find a valid v in avail
            # We can iterate through avail or pick random
            # Random pick is better to avoid bias
            
            # To avoid O(N) check for "v != u", we just check
            for _ in range(30):
                if u not in avail: # u might have been removed if deg reached k? No, we pick u from avail
                    break
                v = random.choice(avail)
                if v == u:
                    continue
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                    
                # Girth Check
                max_depth = g - 2
                valid = True
                if max_depth >= 1:
                    visited = {u: 0}
                    queue = [u]
                    for d in range(1, max_depth + 1):
                        next_q = []
                        for node in queue:
                            for nei in adj[node]:
                                if nei not in visited:
                                    if nei == v:
                                        valid = False
                                        break
                                    visited[nei] = d
                                    next_q.append(nei)
                        if not valid:
                            break
                        queue = next_q
                        if not queue:
                            break
                
                if valid:
                    deg[u] += 1
                    deg[v] += 1
                    adj[u].append(v)
                    adj[v].append(u)
                    edge_set.add((u, v))
                    edge_set.add((v, u))
                    edges.append((u, v))
                    
                    # Update avail
                    if deg[u] == k:
                        avail.remove(u)
                    if deg[v] == k:
                        avail.remove(v)
                        
                    found_v = True
                    break
                    
            if not found_v:
                success = False
                break
                
        if success and all(d == k for d in deg):
            return edges
            
    return None
