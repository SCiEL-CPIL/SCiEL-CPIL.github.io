import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Fallback: LUW Construction from seed
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # 2. Local Search Strategy
    def random_reg_graph(n, k):
        # Generates a random k-regular graph
        # Using configuration model
        half_edges = n * k
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        for i in range(0, half_edges, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                # Retry
                return random_reg_graph(n, k)
            if v in adj[u]:
                return random_reg_graph(n, k)
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def girth(adj):
        n = len(adj)
        min_girth = float('inf')
        for i in range(n):
            # BFS
            visited = {i: 0}
            queue = [(i, 0)]
            while queue:
                u, dist = queue.pop(0)
                if dist >= min_girth:
                    continue
                for v in adj[u]:
                    if v not in visited:
                        visited[v] = dist + 1
                        queue.append((v, dist + 1))
                    elif dist + 1 > visited[v] and v != i:
                        # Found a cycle
                        cycle_len = dist + 1 + visited[v]
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth <= g:
                            return min_girth
        return min_girth if min_girth != float('inf') else 2*n

    def local_search(adj, n, k, target_girth, max_time):
        best_adj = [s.copy() for s in adj]
        best_girth = girth(best_adj)
        if best_girth >= target_girth:
            return best_adj, best_girth
        
        while time.time() < max_time:
            # Pick two random edges
            # To be efficient, pick random vertices
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or v not in adj[u]:
                continue
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y or y not in adj[x]:
                continue
            if (u == x or u == y or v == x or v == y):
                continue
            
            # Swap (u,v) and (x,y) to (u,x) and (v,y)
            # Check validity
            if x in adj[u] or y in adj[v]:
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
            
            # Check girth periodically
            if random.random() < 0.01:
                cur_girth = girth(adj)
                if cur_girth > best_girth:
                    best_girth = cur_girth
                    best_adj = [s.copy() for s in adj]
                    if best_girth >= target_girth:
                        return best_adj, best_girth

        return best_adj, best_girth

    # Main Logic
    # Calculate Moore Bound
    if g % 2 == 0:
        m = g // 2
        moore = 2 * sum((k-1)**i for i in range(m))
    else:
        m = (g-1)//2
        moore = 1 + k * sum((k-1)**i for i in range(m))
        
    # Start search from Moore bound, but step up
    # For k=4, g=10, moore=242. Start at 320.
    n_start = max(moore, 320) if (k==4 and g==10) else moore
    
    best_n = float('inf')
    best_edges = None
    
    # Try local search for a few values of N
    for N in range(n_start, n_start + 100, 2):
        if time.time() > deadline - 20:
            break
        adj = random_reg_graph(N, k)
        adj, girth_val = local_search(adj, N, k, g, deadline - 10)
        if girth_val >= g:
            # Convert to edges
            edges = []
            for i in range(N):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            if N < best_n:
                best_n = N
                best_edges = edges
            # If we found a very small one, stop
            if N <= moore + 10:
                break

    if best_edges:
        return best_n, best_edges
    else:
        # Fallback to LUW
        return luw_construct(k, g)
