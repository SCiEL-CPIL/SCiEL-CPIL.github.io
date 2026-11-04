import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, 10s margin

    # Fallback: LUW construction (fixed)
    def luw_construct():
        q = k
        d = g - 4  # For g=10, d=6
        if q == 4:
            # XOR-based field GF(4)
            # Elements: 0,1,2,3 with 0=0, 1=1, 2=α, 3=α+1
            # Addition: XOR
            # Multiplication table for GF(4)
            table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y  # In char 2, sub = add
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # LUW rules for d=6 (girth 10)
        # rule[c] = (a, b) meaning y[c] = x[c] ± x[a]*y[b]
        # For d=6:
        rule = {
            2: (1, 1),
            3: (2, 1),
            4: (1, 2),
            5: (3, 2),
            6: (1, 3)
        }

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
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Local Search for k=4, g=10
    if k == 4 and g == 10:
        def get_girth_fast(adj, max_check):
            """Fast girth check - returns girth if < max_check, else max_check"""
            n = len(adj)
            min_girth = max_check
            for start in range(n):
                # BFS with non-backtracking
                dist = [-1] * n
                prev = [-1] * n
                dist[start] = 0
                queue = [start]
                found = False
                while queue:
                    next_queue = []
                    for u in queue:
                        if dist[u] >= max_check:
                            continue
                        for v in adj[u]:
                            if v == prev[u]:
                                continue
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                prev[v] = u
                                next_queue.append(v)
                            else:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < min_girth:
                                    min_girth = cycle_len
                                    if min_girth < max_check:
                                        found = True
                    queue = next_queue
                    if found and min_girth < max_check:
                        break
                if min_girth < max_check:
                    return min_girth
            return min_girth

        def random_4reg(n):
            """Generate random 4-regular graph using configuration model with repair"""
            for _ in range(50):
                half_edges = []
                for i in range(n):
                    for _ in range(4):
                        half_edges.append(i)
                random.shuffle(half_edges)
                adj = [[] for _ in range(n)]
                valid = True
                for i in range(0, len(half_edges), 2):
                    u = half_edges[i]
                    v = half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    if v in adj[u]:
                        valid = False
                        break
                    adj[u].append(v)
                    adj[v].append(u)
                if valid and all(len(adj[i]) == 4 for i in range(n)):
                    return adj
            return None

        best_n = None
        best_edges = None

        # Try n values from 320 down
        for n in range(320, 242, -5):
            if time.time() > deadline:
                break
            found = False
            max_attempts = 50 if n > 280 else 100
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                adj = random_4reg(n)
                if adj is None:
                    continue

                # Local search with 2-switches
                for step in range(2000):
                    if time.time() > deadline:
                        break
                    girth = get_girth_fast(adj, 10)
                    if girth >= 10:
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        best_n = n
                        best_edges = edges
                        found = True
                        break

                    # Find a short cycle and break it
                    # Pick random edge, try 2-switch
                    u = random.choice(range(n))
                    v = random.choice(adj[u])

                    # Find non-adjacent vertex
                    candidates = [x for x in range(n) if x != u and x != v and x not in adj[u]]
                    if not candidates:
                        continue
                    x = random.choice(candidates)
                    y = random.choice(adj[x])
                    if y == u or y == v or x in adj[u] or y in adj[v]:
                        continue

                    # Check degrees won't be violated
                    if x == v or y == u:
                        continue

                    # Perform 2-switch
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)

                if found:
                    break
            if found:
                break

        if best_n is not None:
            return best_n, best_edges

    return luw_construct()
