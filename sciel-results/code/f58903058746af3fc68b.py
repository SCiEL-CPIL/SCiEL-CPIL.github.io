"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 130.0
    random.seed(42)

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def smallest_prime_ge(n):
        if n <= 2:
            return 2
        p = n if n % 2 != 0 else n + 1
        while not is_prime(p):
            p += 2
        return p

    def build_luw(q, g):
        """Build LUW graph CD(d,q) with girth >= g. Returns (n, edges, adjacency)."""
        if g <= 5:
            d = 3
        else:
            d = 3
            while True:
                girth_est = d + 5 if d % 2 == 1 else d + 4
                if girth_est >= g:
                    break
                d += 1

        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add_op = lambda x, y: x ^ y
            sub_op = lambda x, y: x ^ y
            mul_op = lambda x, y: table[x][y]
        else:
            add_op = lambda x, y: (x + y) % q
            sub_op = lambda x, y: (x - y) % q
            mul_op = lambda x, y: x * y % q

        add_t = [[add_op(i, j) for j in range(q)] for i in range(q)]
        sub_t = [[sub_op(i, j) for j in range(q)] for i in range(q)]
        mul_t = [[mul_op(i, j) for j in range(q)] for i in range(q)]

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
        for c in range(2, d + 1):
            rule_a[c], rule_b[c] = rule[c]

        def compute_neighbors_point(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add_t[x[c]][mul_t[y[a]][x[b]]]
                result.append((1, tuple(y)))
            return result

        def compute_neighbors_line(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub_t[x[c]][mul_t[x[a]][y[b]]]
                result.append((0, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []

        queue = deque([start])
        while queue:
            if time.time() > deadline:
                break
            vertex = queue.popleft()
            side, x = vertex
            if side == 0:
                neighbors = compute_neighbors_point(x)
            else:
                neighbors = compute_neighbors_line(x)
            for other in neighbors:
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                    queue.append(other)
                edges.append((index[vertex], index[other]))

        n = len(order)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        return n, edges, adj

    def check_girth_fast(adj, n, g):
        max_depth = (g - 1) // 2
        for start in range(n):
            if time.time() > deadline:
                return True
            visited = [-1] * n
            visited[start] = 0
            queue = deque([(start, -1, 0)])
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if visited[nb] == -1:
                        visited[nb] = depth + 1
                        queue.append((nb, node, depth + 1))
                    else:
                        cycle_len = depth + 1 + visited[nb]
                        if cycle_len < g:
                            return False
        return True

    q = smallest_prime_ge(k)
    n, edges, adj = build_luw(q, g)

    if not check_girth_fast(adj, n, g):
        return n, edges

    if q == k:
        return n, edges

    if is_prime(k):
        n2, edges2, adj2 = build_luw(k, g)
        if check_girth_fast(adj2, n2, g):
            return n2, edges2

    return n, edges
# EVOLVE-BLOCK-END
