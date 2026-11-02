import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
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

    # Try to find a smaller circulant graph
    def check_girth_circulant(n, generators, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for s in generators:
                j1 = (i + s) % n
                j2 = (i - s) % n
                adj[i].append(j1)
                adj[i].append(j2)
        
        # Girth check via BFS from each node is O(n^2). 
        # But for circulant, BFS from 0 is sufficient?
        # No, a cycle might not pass through 0.
        # However, for a circulant graph, if there is a cycle of length L,
        # there is a word of length L in the generators summing to 0.
        # So we just need to check if there is a non-trivial word of length < g summing to 0 mod n.
        # This is equivalent to BFS from 0 in the Cayley graph, checking for collisions.
        # If a node is reached by two different paths of length d1, d2, 
        # and d1 + d2 < g, we have a cycle?
        # Actually, if we reach node v with dist[v] = d, and we find an edge to v from u with dist[u] + 1 = d',
        # and d' + d < g, we have a cycle.
        # BFS from 0 finds the shortest distance to all nodes.
        # If the graph is connected, this covers all "shortest" cycles through 0.
        # What about cycles not through 0?
        # In a Cayley graph, any cycle can be translated to pass through 0?
        # Yes, if the graph is vertex-transitive (Cayley graphs are).
        # So checking BFS from 0 is sufficient to find the girth.
        
        dist = [-1] * n
        dist[0] = 0
        queue = deque([0])
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
        return True

    # Search for circulant
    if k == 3 and g == 16:
        # Try n from 512 upwards
        # We will try a few n values.
        # To save time, we can skip n that are unlikely.
        # But let's just iterate.
        
        # Pre-generate some random seeds? No, random is fine.
        
        for n in range(512, 1280):
            if time.time() > deadline:
                break
            
            # Determine tries based on remaining time
            # Rough estimate: 1 check ~ 1ms.
            # 110s / 1ms = 110,000 checks.
            # 768 n values.
            # ~140 checks per n.
            
            tries = 100
            for _ in range(tries):
                if time.time() > deadline:
                    break
                
                gen = set()
                while len(gen) < 3:
                    s = random.randint(1, n - 1)
                    if s == 0: continue
                    conflict = False
                    for existing in gen:
                        if s == existing or s == n - existing:
                            conflict = True
                            break
                    if not conflict:
                        gen.add(s)
                
                if len(gen) < 3:
                    continue
                    
                gen_list = list(gen)
                
                if check_girth_circulant(n, gen_list, g):
                    edges = []
                    for i in range(n):
                        for s in gen_list:
                            edges.append((i, (i + s) % n))
                    return n, edges

    # Fallback
    return luw_construct(k, g)
