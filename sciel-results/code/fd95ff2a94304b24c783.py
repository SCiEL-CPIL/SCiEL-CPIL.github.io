import time
import heapq

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    q = k - 1
    if q < 2:
        q = 2
        
    base_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    best_n = float('inf')
    best_edges = []
    
    for d in range(base_d, base_d + 5): # Try up to 5 more depths
        if time.time() > deadline - 10:
            break
            
        # Setup field ops
        if q == 3:
            add = lambda x, y: (x + y) % 3
            sub = lambda x, y: (x - y) % 3
            mul = lambda x, y: x * y % 3
        elif q == 4:
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
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        
        for vertex in order:
            if time.time() > deadline - 5:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        n = len(order)
        if n >= best_n:
            continue
            
        # Verify Girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        def has_short_cycle(girth_limit):
            for u, v in edges:
                # Shortest path from u to v avoiding (u,v)
                dist = [float('inf')] * n
                dist[u] = 0
                pq = [(0, u)]
                while pq:
                    d, cur = heapq.heappop(pq)
                    if d > dist[cur]: continue
                    if cur == v:
                        return d + 1 < girth_limit
                    for w in adj[cur]:
                        if (cur == u and w == v) or (cur == v and w == u):
                            continue
                        if dist[w] > d + 1:
                            dist[w] = d + 1
                            heapq.heappush(pq, (d+1, w))
                if dist[v] < float('inf'):
                    return dist[v] + 1 < girth_limit
            return False

        if not has_short_cycle(g):
            best_n = n
            best_edges = edges
            # We can stop if we hit the Moore bound?
            # Moore bound for 4,10 is 485.
            # If n == 486, it's optimal (likely).
            break
            
    if best_n == float('inf'):
        # Fallback to original q=k logic if all failed?
        # Or just return the last tried one?
        pass

    return best_n, best_edges
