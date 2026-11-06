import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, n, remaining_edges):
        best_len = 0
        best_cycle = []
        
        for start in range(n):
            if not adj[start]:
                continue
            
            visited = {start: 0}
            parent = {start: None}
            in_stack = {start}
            
            def dfs(u):
                nonlocal best_len, best_cycle
                for v in sorted(adj[u]):
                    if (min(u, v), max(u, v)) not in remaining_edges:
                        continue
                    if v not in visited:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        in_stack.add(v)
                        dfs(v)
                        in_stack.remove(v)
                    elif v in in_stack:
                        length = visited[u] - visited[v] + 1
                        if length >= 3 and length > best_len:
                            cycle = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                cycle.append(cur)
                            best_cycle = cycle
                            best_len = length

            dfs(start)
            
            if time.time() > deadline:
                break

        return best_cycle

    def min_t_join(n, edges, odd_vertices):
        if not odd_vertices:
            return set()
        
        # Compute shortest paths between all odd vertices
        # Use Floyd-Warshall for simplicity since n <= 30
        INF = float('inf')
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        
        # Build adjacency matrix for distances
        for u, v in edges:
            dist[u][v] = 1
            dist[v][u] = 1
            
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Now we need min weight perfect matching on odd_vertices
        # Map odd vertices to indices 0..m-1
        m = len(odd_vertices)
        if m % 2 != 0:
            # This should not happen in a graph (sum of degrees is even)
            return set()
            
        # Cost matrix for matching
        cost = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                if i == j:
                    cost[i][j] = 0
                else:
                    cost[i][j] = dist[odd_vertices[i]][odd_vertices[j]]
                    
        # DP for perfect matching if m <= 16
        if m <= 16:
            dp = [INF] * (1 << m)
            dp[0] = 0
            # Store parent for reconstruction? Or just find the matching edges.
            # We need the actual edges in the join.
            # It's easier to just find the matching pairs and then reconstruct paths.
            
            # We need to know which pairs were matched.
            # Let's store the choice.
            choice = [None] * (1 << m)
            
            for mask in range(1 << m):
                if dp[mask] == INF:
                    continue
                # Find first unset bit
                i = 0
                while i < m and (mask & (1 << i)):
                    i += 1
                if i == m:
                    continue
                # Try to match i with j
                for j in range(i + 1, m):
                    if not (mask & (1 << j)):
                        new_mask = mask | (1 << i) | (1 << j)
                        new_cost = dp[mask] + cost[i][j]
                        if new_cost < dp[new_mask]:
                            dp[new_mask] = new_cost
                            choice[new_mask] = (i, j)
                            
            full_mask = (1 << m) - 1
            if dp[full_mask] == INF:
                # Fallback to greedy
                pass
            else:
                # Reconstruct matching
                matching = []
                mask = full_mask
                while mask:
                    i, j = choice[mask]
                    matching.append((odd_vertices[i], odd_vertices[j]))
                    mask ^= (1 << i) | (1 << j)
                
                # Now find the edges in the T-Join
                # The T-Join is the symmetric difference of the shortest paths for each pair in matching.
                # We need the actual paths.
                # Re-run Dijkstra or use parent pointers from Floyd-Warshall?
                # Floyd-Warshall doesn't store parents easily.
                # Let's just run Dijkstra from each source in odd_vertices to get parents.
                
                # Actually, we can just use the dist matrix to find the path?
                # No, we need the specific path edges.
                # Let's run Dijkstra for each source in odd_vertices.
                
                # Since n is small, we can just do BFS for each pair in matching.
                # Or, simpler:
                # For each pair (u, v) in matching, find a shortest path in G.
                # Add edges of path to a multiset.
                # The T-Join edges are those with odd count.
                
                edge_counts = {}
                
                for u, v in matching:
                    # BFS to find path
                    parent = {u: None}
                    queue = [u]
                    while queue:
                        curr = queue.pop(0)
                        if curr == v:
                            break
                        for nei, _ in [(nei, 1) for nei in range(n) if dist[curr][nei] == 1]:
                            if nei not in parent:
                                parent[nei] = curr
                                queue.append(nei)
                    
                    # Reconstruct path
                    path = []
                    curr = v
                    while curr != u:
                        prev = parent[curr]
                        path.append((min(prev, curr), max(prev, curr)))
                        curr = prev
                    
                    for e in path:
                        edge_counts[e] = edge_counts.get(e, 0) + 1
                        
                t_join_edges = {e for e, c in edge_counts.items() if c % 2 == 1}
                return t_join_edges
        
        # Fallback: Greedy matching
        # Sort all pairs by cost
        pairs = []
        for i in range(m):
            for j in range(i+1, m):
                pairs.append((cost[i][j], i, j))
        pairs.sort()
        
        matched = [False] * m
        matching = []
        for c, i, j in pairs:
            if not matched[i] and not matched[j]:
                matching.append((odd_vertices[i], odd_vertices[j]))
                matched[i] = True
                matched[j] = True
        
        edge_counts = {}
        for u, v in matching:
            # BFS for path
            parent = {u: None}
            queue = [u]
            while queue:
                curr = queue.pop(0)
                if curr == v:
                    break
                for nei in range(n):
                    if dist[curr][nei] == 1 and nei not in parent:
                        parent[nei] = curr
                        queue.append(nei)
            
            path = []
            curr = v
            while curr != u:
                prev = parent[curr]
                path.append((min(prev, curr), max(prev, curr)))
                curr = prev
            
            for e in path:
                edge_counts[e] = edge_counts.get(e, 0) + 1
                
        t_join_edges = {e for e, c in edge_counts.items() if c % 2 == 1}
        return t_join_edges

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        n = spec[0]
        
        # Build adjacency and degree
        deg = [0] * n
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            
        odd_vertices = [i for i in range(n) if deg[i] % 2 == 1]
        
        # Find min T-Join
        t_join = min_t_join(n, edges, odd_vertices)
        
        # H is edges not in t_join
        h_edges = [e for e in edges if e not in t_join]
        
        # Decompose H into cycles
        h_adj = {i: set() for i in range(n)}
        for u, v in h_edges:
            h_adj[u].add(v)
            h_adj[v].add(u)
            
        cycles = []
        remaining_h = set(h_edges)
        
        while remaining_h:
            if time.time() > deadline:
                # Add remaining as singles
                for e in remaining_h:
                    cycles.append(list(e))
                break
                
            best_cycle = find_longest_cycle(h_adj, n, remaining_h)
            if best_cycle:
                cycles.append(best_cycle)
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining_h:
                        remaining_h.discard(e)
                        h_adj[u].remove(v)
                        h_adj[v].remove(u)
            else:
                for e in remaining_h:
                    cycles.append(list(e))
                break
                
        # Add T-Join edges as single edges
        for e in t_join:
            cycles.append(list(e))
            
        results.append(cycles)
        
    return results
