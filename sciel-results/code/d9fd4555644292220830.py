import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # If specific case, try to find a better graph
    if k == 4 and g == 10:
        # Moore bound for k=4, g=10 is 682
        min_n = 682
        max_n = 1500
        
        for n in range(min_n, max_n + 1):
            if time.time() > deadline:
                break
            # Try circulant graphs C_n({a, -a, b, -b})
            # a, b must be such that the graph is simple
            # and girth >= 10
            
            found = False
            # Limit search space for a, b
            # a, b < n/2
            for a in range(1, n // 2 + 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n // 2 + 1):
                    # Check simplicity: a != b, a != -b, etc.
                    # In Z_n, a != b, a != n-b, b != n-a
                    if a == b or a == n - b or b == n - a:
                        continue
                    if 2 * a % n == 0 or 2 * b % n == 0:
                         # Loop if 2a=0? No, loop if a=0 or a=n/2.
                         # If n is even, a=n/2 gives loop.
                         if a == n // 2 or b == n // 2:
                             continue
                    
                    # Check girth
                    if _check_girth_circulant(n, a, b, g):
                        # Construct edges
                        edges = []
                        for i in range(n):
                            edges.append((i, (i + a) % n))
                            edges.append((i, (i - a) % n))
                            edges.append((i, (i + b) % n))
                            edges.append((i, (i - b) % n))
                        # Remove duplicates and loops
                        unique_edges = set()
                        for u, v in edges:
                            if u == v: continue
                            if u > v: u, v = v, u
                            unique_edges.add((u, v))
                        return n, list(unique_edges)
            
            if found:
                break
        
        # If no circulant found, fall back to LUW
        pass

    # Fallback: LUW construction (same as seed)
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

def _check_girth_circulant(n, a, b, g):
    # Check if C_n({a, -a, b, -b}) has girth >= g
    # Girth is min cycle length.
    # We can check for cycles of length 3 to g-1.
    # For small g (10), we can just do BFS up to depth 4.
    # If no collision in BFS tree, girth > 2*4 = 8?
    # Actually, if BFS to depth 4 finds no back edge, girth >= 9.
    # We need girth >= 10.
    # So we need to ensure no cycle of length 9 either.
    # BFS to depth 4 covers paths of length 4.
    # A cycle of length 9 implies a path of length 4 from u to v and path of length 5 from v to u?
    # Simpler: Just count the number of closed walks of length l < g.
    # Or just BFS from 0 to depth 4. If no cycle found, check depth 5?
    # For n ~ 700, BFS is fast.
    
    # BFS from 0
    # dist[v] = min distance from 0
    # parent[v]
    # If we find a back edge, calculate cycle length.
    
    from collections import deque
    
    # We need to check ALL vertices?
    # In a vertex-transitive graph (circulant), checking from one vertex is not enough for girth?
    # Actually, girth is the minimum over all vertices.
    # But if there is a short cycle, it will be found in BFS from any vertex in the cycle.
    # So we can just check if ANY vertex has a short cycle.
    # However, for circulant graphs, the structure is uniform.
    # If a cycle of length L exists, it exists "everywhere" in a sense.
    # But to be safe, we can check a few vertices.
    
    # Actually, if the girth is < g, there is a cycle of length L < g.
    # We can just check if the number of closed walks of length L is 0 for L < g.
    # This is expensive.
    
    # Let's just do BFS from vertex 0 to depth g//2.
    # If we find a back edge to a vertex at depth d, cycle length is depth[u] + depth[v] + 1.
    # If min cycle length >= g, return True.
    
    max_depth = g // 2
    
    for start_v in range(0, min(5, n)): # Check a few starts
        dist = [-1] * n
        parent = [-1] * n
        dist[start_v] = 0
        queue = deque([start_v])
        
        min_cycle = float('inf')
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in [(u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Back edge
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
        
        if min_cycle < g:
            return False
            
    return True
