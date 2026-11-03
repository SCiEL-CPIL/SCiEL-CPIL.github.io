import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def luw_fallback():
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

    if k == 5 and g == 9:
        try:
            # Structural change: use deterministic algebraic construction via
            # PSL(2, q) with carefully selected generators, not random search.
            # For k=5, use q=13, PSL(2,13) has order 1092.
            # The Cayley graph with 5 involutions can be 5-regular if we use
            # a generating set of 5 elements of order 2.
            
            q = 13
            # Build PSL(2, q) elements as (a,b,c,d) with ad-bc=1, modulo ±I
            sl_elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a != 0:
                            d_val = (1 + b * c) * pow(a, -1, q) % q
                            sl_elements.append((a, b, c, d_val))
                        else:
                            if b == 0:
                                continue
                            # ad - bc = 1 => -bc = 1 => c = -b^{-1}
                            c_val = (-1 * pow(b, -1, q)) % q
                            if c_val != c:
                                continue
                            for d_val in range(q):
                                sl_elements.append((a, b, c, d_val))
            
            # Deduplicate modulo ±I
            psl_elements = []
            seen = set()
            for e in sl_elements:
                a, b, c, d_val = e
                key = (a, b, c, d_val)
                neg_key = (-a % q, -b % q, -c % q, -d_val % q)
                if key < neg_key:
                    if key not in seen:
                        seen.add(key)
                        psl_elements.append(e)
                else:
                    if neg_key not in seen:
                        seen.add(neg_key)
                        psl_elements.append(neg_key)
            
            n_vertices = len(psl_elements)
            idx_map = {e: i for i, e in enumerate(psl_elements)}
            
            # Multiplication table
            mul_table = np.zeros((n_vertices, n_vertices), dtype=np.int32)
            for i in range(n_vertices):
                a1, b1, c1, d1 = psl_elements[i]
                for j in range(n_vertices):
                    a2, b2, c2, d2 = psl_elements[j]
                    prod = ((a1 * a2 + b1 * c2) % q,
                            (a1 * b2 + b1 * d2) % q,
                            (c1 * a2 + d1 * c2) % q,
                            (c1 * b2 + d1 * d2) % q)
                    mul_table[i, j] = idx_map[prod]
            
            # Find all involutions (order 2 elements)
            involution_indices = []
            for i, e in enumerate(psl_elements):
                a, b, c, d_val = e
                if (a + d_val) % q == 0 and (b, c) != (0, 0):
                    involution_indices.append(i)
            
            involution_indices = np.array(involution_indices, dtype=np.int32)
            n_inv = len(involution_indices)
            
            # Use a deterministic set of 5 involutions that are known to give
            # high girth. Use the first 5 in a canonical order.
            # To maximize girth, use involutions that are "far apart" in the group.
            
            # Try the first 5 involutions
            S_idx = involution_indices[:5]
            
            # Build the Cayley graph
            visited = np.zeros(n_vertices, dtype=bool)
            visited[0] = True
            queue = [0]
            head = 0
            edges = []
            while head < len(queue):
                u = queue[head]
                head += 1
                for s in range(5):
                    v = mul_table[u, S_idx[s]]
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
                        edges.append((u, v))
            
            if not visited.all():
                return luw_fallback()
            
            # Check girth using BFS from each vertex
            def check_girth(n, edges, g_target):
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                half_g = g_target // 2
                for s in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[s] = 0
                    q = deque([s])
                    while q:
                        u = q.popleft()
                        if dist[u] >= half_g:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                q.append(v)
                            else:
                                if v != parent[u]:
                                    cycle_len = dist[u] + dist[v] + 1
                                    if cycle_len < g_target:
                                        return False
                return True
            
            if check_girth(n_vertices, edges, g):
                return n_vertices, edges
            else:
                return luw_fallback()
                
        except Exception:
            pass

    return luw_fallback()
