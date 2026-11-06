<<<<<<< SEARCH
def _greedy_decompose(edges, deadline):
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    degree = defaultdict(int)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1
        
    paths = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Pick start vertex: prefer odd degree, then high degree
        candidates = [v for v in adj if adj[v]]
        if not candidates:
            break
            
        odd_verts = [v for v in candidates if degree.get(v, 0) % 2 == 1]
        if odd_verts:
            # Random choice among odd vertices with some bias
            weights = [degree[v] for v in odd_verts]
            total = sum(weights)
            r = random.random() * total
            cum = 0
            start = odd_verts[0]
            for v, w in zip(odd_verts, weights):
                cum += w
                if r <= cum:
                    start = v
                    break
        else:
            # Pick highest degree vertex with some randomness
            candidates.sort(key=lambda x: (-degree.get(x, 0), random.random()))
            start = candidates[0]
            
        # Find long path using Warnsdorff with randomization
        path = [start]
        visited = {start}
        current = start
        
        while True:
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break
            
            # Warnsdorff: prefer neighbors with fewer unvisited neighbors
            scored = []
            for n in neighbors:
                unvisited_count = sum(1 for y in adj[n] if y not in visited)
                # Add small random perturbation
                scored.append((unvisited_count + random.random() * 0.5, n))
            scored.sort()
            next_v = scored[0][1]
            
            path.append(next_v)
            visited.add(next_v)
            
            # Remove edge
            edge = tuple(sorted((current, next_v)))
            remaining.discard(edge)
            adj[current].discard(next_v)
            adj[next_v].discard(current)
            degree[current] = degree.get(current, 0) - 1
            degree[next_v] = degree.get(next_v, 0) - 1
            
            if not adj[current]:
                del adj[current]
            if not adj[next_v]:
                del adj[next_v]
            
            current = next_v
            
        paths.append(path)
        
    return paths
=======
def _greedy_decompose(edges, deadline):
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    degree = defaultdict(int)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1
        
    paths = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Pick start vertex: prefer odd degree, then high degree
        candidates = [v for v in adj if adj[v]]
        if not candidates:
            break
            
        odd_verts = [v for v in candidates if degree.get(v, 0) % 2 == 1]
        if odd_verts:
            # Random choice among odd vertices with some bias
            weights = [degree[v] for v in odd_verts]
            total = sum(weights)
            r = random.random() * total
            cum = 0
            start = odd_verts[0]
            for v, w in zip(odd_verts, weights):
                cum += w
                if r <= cum:
                    start = v
                    break
        else:
            # Pick highest degree vertex with some randomness
            candidates.sort(key=lambda x: (-degree.get(x, 0), random.random()))
            start = candidates[0]
            
        # Find long path using Warnsdorff with randomization
        path = [start]
        visited = {start}
        current = start
        
        while True:
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break
            
            # Warnsdorff: prefer neighbors with fewer unvisited neighbors
            scored = []
            for n in neighbors:
                unvisited_count = sum(1 for y in adj[n] if y not in visited)
                # Add small random perturbation
                scored.append((unvisited_count + random.random() * 0.5, n))
            scored.sort()
            next_v = scored[0][1]
            
            path.append(next_v)
            visited.add(next_v)
            
            # Remove edge
            edge = tuple(sorted((current, next_v)))
            remaining.discard(edge)
            adj[current].discard(next_v)
            adj[next_v].discard(current)
            degree[current] = degree.get(current, 0) - 1
            degree[next_v] = degree.get(next_v, 0) - 1
            
            if not adj[current]:
                del adj[current]
            if not adj[next_v]:
                del adj[next_v]
            
            current = next_v
            
        paths.append(path)
        
    return paths
>>>>>>> REPLACE

Now let me improve the edge-end pairing with a better strategy that uses local search on the pairing itself:

