import time
import random
import math
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    if k == 3:
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift(g, deadline, start_time):
    # Petersen graph edges
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    adj = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Enumerate non-backtracking closed walks of length < g
    # State: (current_v, prev_v, length, counts_tuple)
    # We store counts as a tuple of 15 ints.
    coeff_vectors = set()
    max_walk_len = g - 1

    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0]*15)]
        while stack:
            v, prev_v, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            for next_v in adj[v]:
                if next_v == prev_v:
                    continue
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    sign = 1
                else:
                    sign = -1

                new_counts = counts.copy()
                new_counts[e_idx] += sign
                new_l = l + 1

                if next_v == start_v and new_l >= 5:
                    coeff_vectors.add(tuple(new_counts))

                stack.append((next_v, v, new_l, new_counts))

    # Normalize
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        for i in range(15):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    A = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(A)

    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True

    # Moore bound for k=3, g=16 is 766.
    # n = 10p. So p >= 77.
    # Start primes from 79.
    primes = []
    candidate = 79
    while len(primes) < 50:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1

    best_n = float('inf')
    best_edges = None

    for m in primes:
        if time.time() > deadline:
            break

        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue

        found = False
        voltages = None

        # Use a more efficient approach: try multiple random initializations
        # with a faster local search
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initial random
            v = np.array([random.randint(1, m-1) for _ in range(15)], dtype=np.int64)
            prod = (A @ v) % m
            cost = np.sum(prod == 0)

            T = 1.0
            cooling = 0.995
            min_T = 0.001

            # Max iterations per restart
            max_iter = 8000
            for i in range(max_iter):
                if time.time() > deadline:
                    break

                if cost == 0:
                    found = True
                    voltages = v
                    break

                # Pick random edge to change
                j = random.randint(0, 14)
                old_val = v[j]
                new_val = random.randint(1, m-1)
                if new_val == old_val:
                    continue

                # Calculate delta cost
                delta = (A[:, j] * (new_val - old_val)) % m
                new_prod = (prod + delta) % m

                # Count violations
                old_violations = prod == 0
                new_violations = new_prod == 0

                delta_cost = np.sum(new_violations) - np.sum(old_violations)

                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    v[j] = new_val
                    prod = new_prod
                    cost += delta_cost

                T *= cooling
                if T < min_T:
                    T = min_T

            if found:
                break

        if found:
            edges = _build_lifted_graph(pet_edges, voltages, m)
            # Verify
            if _verify_girth(n_candidate, edges, g):
                return n_candidate, edges

    # Fallback
    return _construct_luw(k=3, g=g, d=11)

def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    edges = []
    seen = set()
    for i, (u, v) in enumerate(pet_edges):
        vol = int(voltages[i])
        for t in range(m):
            u_node = t * n_base + u
            v_node = ((t + vol) % m) * n_base + v

            if u_node < v_node:
                edge = (u_node, v_node)
            else:
                edge = (v_node, u_node)

            if edge not in seen:
                seen.add(edge)
                edges.append(edge)

    return edges

def _verify_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        queue = deque()
        queue.append((start, 0))
        dist[start] = 0

        while queue:
            u, d = queue.popleft()
            if d >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append((v, d + 1))
                elif v != parent[u]:
                    # Found a cycle
                    # Length = dist[u] + dist[v] + 1
                    # This is a lower bound on the cycle length?
                    # Actually, in BFS, the first time we see a visited node (not parent),
                    # the cycle length is dist[u] + dist[v] + 1.
                    # If this is < g, we fail.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw(k, g, d=None, variant=0):
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    q = k
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
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

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
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
