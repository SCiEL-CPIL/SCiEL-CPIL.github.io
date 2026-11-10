import random
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

    # Attempt to find a smaller graph using Circulant Search
    start_time = time.time()
    timeout = 110 # Leave 10s for fallback/verification
    
    # Moore bound
    if g % 2 == 0:
        r = g // 2
        moore = 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g-1)//2
        moore = 1 + k * sum((k-1)**i for i in range(r))
        
    # Try circulant graphs
    found = False
    best_n = None
    best_edges = None
    
    # Search N from moore to 320 (known cage size)
    for n in range(moore, 321):
        if time.time() - start_time > timeout:
            break
            
        # Try random circulant graphs
        for _ in range(2000): # Increased samples
            if time.time() - start_time > timeout:
                break
                
            s1 = random.randint(1, n//2)
            s2 = random.randint(1, n//2)
            
            if s1 == s2 or s1 == n - s2 or s2 == n - s1 or s1 == 0 or s2 == 0:
                continue
            if s1 == n - s1 or s2 == n - s2:
                continue
                
            # Check girth using BFS from 0
            # For vertex-transitive graphs, girth is the shortest cycle through 0
            visited = [-1] * n
            parent = [-1] * n
            visited[0] = 0
            queue = deque([0])
            girth_found = n 
            
            while queue:
                u = queue.popleft()
                if visited[u] >= g//2:
                    continue # Don't expand further if we can't form a cycle < g
                
                for s in [s1, -s1, s2, -s2]:
                    v = (u + s) % n
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = visited[u] + visited[v] + 1
                        if cycle_len < girth_found:
                            girth_found = cycle_len
                        if girth_found < g:
                            break
                if girth_found < g:
                    break
            
            if girth_found >= g:
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
