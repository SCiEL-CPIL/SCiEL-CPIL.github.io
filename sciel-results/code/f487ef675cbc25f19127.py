import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # safety margin for 150s limit

    # LUW graph construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def check_degree(adj, k):
        return all(len(a) == k for a in adj)

    def girth_of_graph(n, adj, k):
        if not check_degree(adj, k):
            return 0
        best = float('inf')
        for s in range(n):
            if time.time() > deadline:
                break
            dist = {}
            dq = deque()
            for nb in adj[s]:
                state = (nb, s)
                dist[state] = 1
                dq.append(state)
            while dq:
                curr, prev = dq.popleft()
                d = dist[(curr, prev)]
                if curr == s:
                    cycle_len = d + 1
                    if cycle_len < best:
                        best = cycle_len
                    if best <= 3:
                        return best
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    state = (nb, curr)
                    if state not in dist:
                        dist[state] = d + 1
                        dq.append(state)
        return best

    # Voltage graph lift: K_{3,3} base with Z_m voltages
    # K_{3,3} edges: (0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)
    # 4-cycles in K_{3,3} (edge index sequences):
    cycles_4 = [
        [0, 3, 4, 1], [0, 3, 5, 2], [0, 6, 7, 1],
        [0, 6, 8, 2], [1, 4, 5, 2], [1, 7, 8, 2],
        [3, 6, 7, 4], [3, 6, 8, 5], [4, 7, 8, 5]
    ]
    # 6-cycles in K_{3,3}:
    cycles_6 = [
        [0, 3, 5, 8, 7, 1], [0, 6, 8, 5, 4, 1],
        [1, 4, 3, 6, 8, 2], [1, 7, 6, 3, 5, 2],
        [2, 5, 3, 6, 7, 1], [2, 8, 6, 3, 4, 1]
    ]

    def voltage_lift(m, voltages):
        """Build the lift of K_{3,3} with Z_m voltages."""
        n = 6 * m
        edges = []
        # Left vertices: 0..2 (base), each with m copies: i*m + j
        # Right vertices: 3..5 (base), each with m copies: i*m + j
        # Edge (base_u, base_v) with voltage v: connects u*m+j to v*m+(j+v)%m
        edge_map = [
            (0, 3, voltages[0]), (0, 4, voltages[1]), (0, 5, voltages[2]),
            (1, 3, voltages[3]), (1, 4, voltages[4]), (1, 5, voltages[5]),
            (2, 3, voltages[6]), (2, 4, voltages[7]), (2, 5, voltages[8])
        ]
        for bu, bv, vol in edge_map:
            for j in range(m):
                u = bu * m + j
                v = bv * m + (j + vol) % m
                edges.append((u, v))
        return n, edges

    def check_voltage_constraints(voltages, m):
        """Check all 4 and 6 cycles have non-zero voltage sum."""
        for cycle in cycles_4:
            s = 0
            for i, e in enumerate(cycle):
                # Determine orientation: need to track direction
                # For a cycle, alternate + and - based on bipartite structure
                if i % 2 == 0:
                    s += voltages[e]
                else:
                    s -= voltages[e]
            if s % m == 0:
                return False
        for cycle in cycles_6:
            s = 0
            for i, e in enumerate(cycle):
                if i % 2 == 0:
                    s += voltages[e]
                else:
                    s -= voltages[e]
            if s % m == 0:
                return False
        return True

    # Try voltage graph lifts
    best_result = None
    best_n = float('inf')

    # LUW baseline
    n_luw, edges_luw = luw_construct(k, g)
    adj_luw = build_adj(n_luw, edges_luw)
    if check_degree(adj_luw, k):
        girth_luw = girth_of_graph(n_luw, adj_luw, k)
        if girth_luw >= g:
            best_n = n_luw
            best_result = (n_luw, edges_luw)

    # Try voltage lifts for k=4, g=10
    if k == 4 and g == 10:
        # Moore bound for n(4,10) is 243, so m must satisfy 6m >= 243, m >= 41
        # Try m from 41 upward, random voltage assignments
        for m in range(41, 80):
            if time.time() > deadline - 5:
                break
            if 6 * m >= best_n:
                break  # Can't improve

            found = False
            for restart in range(500):
                if time.time() > deadline - 5:
                    break
                voltages = list(np.random.randint(1, m, size=9))
                if check_voltage_constraints(voltages, m):
                    n_lift, edges_lift = voltage_lift(m, voltages)
                    adj_lift = build_adj(n_lift, edges_lift)
                    if check_degree(adj_lift, k):
                        girth_lift = girth_of_graph(n_lift, adj_lift, k)
                        if girth_lift >= g:
                            if n_lift < best_n:
                                best_n = n_lift
                                best_result = (n_lift, edges_lift)
                                found = True
                    if found:
                        break

    if best_result is not None:
        return best_result

    # Fallback
    return n_luw, edges_luw
