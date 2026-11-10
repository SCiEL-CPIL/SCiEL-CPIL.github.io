import time
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    start_time = time.time()
    timeout = 110  # Leave 10s for LUW construction and safety
    
    # Moore bound
    if g % 2 == 0:
        r = g // 2
        moore = 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g-1)//2
        moore = 1 + k * sum((k-1)**i for i in range(r))
        
    # Systematic search for circulant graphs
    # n from moore to moore + 200 (covering up to 320 for k=4, g=10)
    # Moore for 4,10 is 2*(1+3+9+27+81) = 2*121 = 242.
    # So we search 242 to 320+.
    
    max_n = moore + 200
    
    for n in range(moore, max_n + 1):
        if time.time() - start_time > timeout:
            break
            
        # s1, s2 are steps. 1 <= s1 < s2 <= n//2
        # Must ensure s1 != s2, s1 != n-s2, etc.
        # For simple graph:
        # s1, s2 must be distinct.
        # s1 != n - s2.
        # If n is even, s != n/2 to maintain degree 4 (distinct steps).
        
        for s1 in range(1, n // 2 + 1):
            for s2 in range(s1 + 1, n // 2 + 1):
                # Check validity
                if n % 2 == 0:
                    if s1 == n // 2 or s2 == n // 2:
                        continue
                if s1 + s2 == n:
                    continue
                
                # Check Girth
                # BFS from 0
                visited = [-1] * n
                parent = [-1] * n
                visited[0] = 0
                queue = deque([0])
                valid = True
                
                while queue:
                    u = queue.popleft()
                    if visited[u] >= g // 2:
                        continue
                    for s in [s1, -s1, s2, -s2]:
                        v = (u + s) % n
                        if visited[v] == -1:
                            visited[v] = visited[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif parent[u] != v and parent[v] != u:
                            # Cycle found
                            cycle_len = visited[u] + visited[v] + 1
                            if cycle_len < g:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    # Construct edges
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + s1) % n))
                        edges.append((i, (i + s2) % n))
                    
                    unique_edges = set()
                    for i in range(n):
                        for j in [(i + s1) % n, (i + s2) % n]:
                            if i < j:
                                unique_edges.add((i, j))
                            else:
                                unique_edges.add((j, i))
                    
                    return n, list(unique_edges)

    # If not found, fallback to LUW
    return luw_construct(k, g)
