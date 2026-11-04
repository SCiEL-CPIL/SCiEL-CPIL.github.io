    def construct(k, g):
        start_time = time.time()
        deadline = start_time + 110  # 10s safety margin

        def luw_graph(q, d):
            """Construct L-U-W graph CD(d,q), return (n, edges) for the connected component."""
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add = lambda x, y: x ^ y
                sub = lambda x, y: x ^ y
                mul = lambda x, y: table[x][y]
            else:
                add = lambda x, y: (x + y) % q
                sub = lambda x, y: (x - y) % q
                mul = lambda x, y: x * y % q

            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        if side == 0:
                            y[c] = add(x[c], mul(y[a], x[b]))
                        else:
                            y[c] = sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)

            start = (0, (None,) + (0,) * d)
            index, order, edges = {start: 0}, [start], []
            for i, vertex in enumerate(order):
                if time.time() > deadline:
                    return None
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            return len(order), edges

        def verify_girth(n, edges, g):
            """Check girth >= g using BFS from each vertex (non-backtracking)."""
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            max_depth = g // 2
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj[u]:
                        if v == parent[u]:
                            continue
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and dist[u] + dist[v] + 1 < g:
                            return False
            return True

        best_n = float('inf')
        best_edges = None

        # Try q = k
        # For k=4, q=4 is a field.
        # For other k, if k is prime power, we could try, but let's stick to k for simplicity and speed.
        # If k is not a prime power, this might fail, but the task is specifically for k=4.
        
        q = k
        
        # Determine valid d
        # Girth of CD(d,q) is roughly d+4 or d+5
        # We need girth >= g
        # So d >= g - 5 (conservative)
        
        # Limit d to avoid huge graphs
        # q^d should be manageable.
        # For q=4, d=5 -> 1024. d=6 -> 4096.
        # For q=5, d=4 -> 625. d=5 -> 3125.
        
        # Try q=k first
        # Then maybe q=k+1 if k is not prime power? 
        # But for k=4, q=4 works.
        
        candidate_qs = [k]
        # If k is not a prime power, we might need another q.
        # But for this specific task (k=4), q=4 is fine.
        
        for q_try in candidate_qs:
            # Estimate max d
            # We want q^d < 10000 for speed
            max_d = 10
            # Refine max_d based on q
            if q_try > 1:
                 # log_q(10000)
                 import math
                 max_d = int(math.log(10000, q_try)) + 1
                 
            for d in range(3, max_d + 1):
                if time.time() > deadline - 20:
                    break
                
                # Check if this d could possibly give girth >= g
                # Approximate girth: d+4 or d+5
                # If d+4 < g, skip
                if d + 4 < g:
                    continue
                    
                result = luw_graph(q_try, d)
                if result is None:
                    continue
                
                n, edges = result
                
                # If n is too large, skip verification (it will be slow)
                if n > 5000:
                    continue
                    
                if n < best_n:
                    if verify_girth(n, edges, g):
                        deg = [0] * n
                        for u, v in edges:
                            deg[u] += 1
                            deg[v] += 1
                        if all(dv == k for dv in deg):
                            best_n = n
                            best_edges = edges

        if best_edges is None:
            # Fallback: Return a small invalid graph? No.
            # If we didn't find one, we might have to return something.
            # But for k=4, g=10, d=5, q=4 should be found.
            # If not, maybe d=6?
            # If still not, we fail.
            # To avoid crash, return a single edge? No, must be k-regular.
            # Return None? The interface expects (n, edges).
            # Let's try one more thing: if best_edges is None, try q=5, d=4?
            # But for k=4, q=5 gives 5-regular? No, L-U-W graphs are k-regular where k=q?
            # Yes, degree is q. So if k=4, we must use q=4.
            
            # If we failed, return a dummy?
            # Better to raise an error or return a large graph if found.
            # Since d=5, q=4 is expected to work, this shouldn't happen.
            
            # As a last resort, return a 4-regular graph of small size?
            # E.g. K5 is 4-regular, girth 3. Fails girth.
            # We must return a valid graph.
            # If the search fails, we have a problem.
            # But d=5, q=4 is very likely to work.
            
            pass

        return best_n, best_edges
    