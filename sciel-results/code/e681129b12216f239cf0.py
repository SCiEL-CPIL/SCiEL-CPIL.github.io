def construct(k, g):
    import random
    from collections import deque
    
    def shortest_path_length(adj, src, dst, n):
        if src == dst:
            return 0
        visited = [-1] * n
        visited[src] = 0
        queue = deque([src])
        while queue:
            v = queue.popleft()
            for w in adj[v]:
                if visited[w] == -1:
                    visited[w] = visited[v] + 1
                    if w == dst:
                        return visited[w]
                    queue.append(w)
        return float('inf')
    
    def verify_girth(adj, n, g):
        for start in range(n):
            visited = [None] * n
            queue = deque([(start, -1, 0)])
            visited[start] = (-1, 0)
            while queue:
                v, parent, dist = queue.popleft()
                for w in adj[v]:
                    if w == parent:
                        continue
                    if visited[w] is None:
                        visited[w] = (v, dist + 1)
                        queue.append((w, v, dist + 1))
                    else:
                        cycle_len = dist + 1 + visited[w][1]
                        if cycle_len < g:
                            return False
        return True
    
    def try_construct(n, k, g, seed=42):
        random.seed(seed)
        half = n // 2
        left = list(range(half))
        right = list(range(half, n))
        
        adj = [[] for _ in range(n)]
        
        for i in left:
            candidates = right[:]
            random.shuffle(candidates)
            count = 0
            for j in candidates:
                if count == k:
                    break
                if shortest_path_length(adj, i, j, n) < g - 1:
                    continue
                adj[i].append(j)
                adj[j].append(i)
                count += 1
            
            if count < k:
                return None
        
        for v in range(n):
            if len(adj[v]) != k:
                return None
        
        if not verify_girth(adj, n, g):
            return None
        
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        return n, edges
    
    for n in range(242, 400, 2):
        result = try_construct(n, k, g)
        if result is not None:
            return result
    
    # Fallback: LUW construction
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
