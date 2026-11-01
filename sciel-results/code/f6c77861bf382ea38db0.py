import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g.
    For k=3, g=16, we use a voltage lift of the Petersen graph.
    """
    start_time = time.time()
    deadline = start_time + 145

    if k != 3:
        return _luw_construct(k, g)

    # Petersen graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
    
    assert len(base_edges) == 15

    random.seed(42)
    best_result = None
    best_n = float('inf')

    # Generate diverse voltage patterns
    def gen_patterns(m, num_patterns):
        """Generate diverse voltage patterns for lift size m."""
        patterns = []
        
        # 1. Linear
        patterns.append([int((i + 1) * m / 15.0) for i in range(15)])
        
        # 2. Golden ratio
        golden = (1 + 5**0.5) / 2.0
        patterns.append([int((i + 1) * m * golden) % m for i in range(15)])
        
        # 3. Prime based
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        patterns.append([primes[i] * m // 15 for i in range(15)])
        
        # 4. Thue-Morse based (good for avoiding small cycles)
        if m > 1:
            tm = [0]
            for _ in range(20):
                tm = tm + [(1 - x) % 2 for x in tm]
            patterns.append([tm[i] * m // 2 for i in range(15)])
        
        # 5. Powers of 2 (mod m)
        patterns.append([pow(2, i, m) for i in range(1, 16)])
        
        # 6. Fibonacci-based
        fib = [1, 1]
        for _ in range(14):
            fib.append((fib[-1] + fib[-2]) % m)
        patterns.append(fib[:15])
        
        # 7. Cubic
        patterns.append([pow(i + 1, 3, m) for i in range(15)])
        
        # 8. Quadratic residue pattern
        if m > 2:
            is_prime = True
            if m % 2 == 0:
                is_prime = (m == 2)
            else:
                for i in range(3, int(m**0.5) + 1, 2):
                    if m % i == 0:
                        is_prime = False
                        break
            if is_prime:
                shifts = []
                for i in range(1, 16):
                    ls = pow(i, (m - 1) // 2, m)
                    shifts.append(m - 1 if ls == m - 1 else 1)
                patterns.append(shifts)
        
        # 9. Random patterns
        for _ in range(num_patterns - len(patterns)):
            patterns.append([random.randint(0, m - 1) for _ in range(15)])
        
        return patterns

    # Search m from 52 upward
    for m in range(52, 1500):
        if time.time() > deadline:
            break
        
        # Number of patterns to try
        if m < 100:
            num_patterns = 100
        elif m < 200:
            num_patterns = 50
        elif m < 500:
            num_patterns = 20
        else:
            num_patterns = 10
        
        patterns = gen_patterns(m, num_patterns)
        
        for shifts in patterns:
            if time.time() > deadline:
                break
            n, edges = _build_lift(base_edges, m, shifts)
            if _verify_girth_fast(n, edges, g):
                return n, edges

    # Fallback: large random lift
    m = 3000
    shifts_rand = [random.randint(0, m - 1) for _ in range(15)]
    n, edges = _build_lift(base_edges, m, shifts_rand)
    return n, edges


def _build_lift(base_edges, m, shifts):
    """Build the lifted graph."""
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using timestamped BFS with early termination.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist = np.full(n, -1, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    seen_mark = np.zeros(n, dtype=np.int32)
    mark = 0

    half_g = g // 2

    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])

        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= half_g:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction for k-regular graphs."""
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    q = 3

    def build_luw(d):
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

    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    return build_luw(11)


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True
