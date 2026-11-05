import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    if k != 3 or g != 16:
        return _luw_construct(k, g)

    # Petersen graph definition
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),   # outer pentagon
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),   # inner pentagram
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)    # spokes
    ]
    n_base = 10
    n_edges_base = 15

    # Build directed voltage lookup structure
    edge_dir = {}  # (u,v) -> edge_index
    for ei, (u, v) in enumerate(petersen_edges):
        edge_dir[(u, v)] = ei
        edge_dir[(v, u)] = ei

    base_adj = [[] for _ in range(n_base)]
    for ei, (u, v) in enumerate(petersen_edges):
        base_adj[u].append(v)
        base_adj[v].append(u)

    def build_derived(voltages, m):
        n_derived = n_base * m
        adj = [[] for _ in range(n_derived)]
        for u in range(n_base):
            for v in base_adj[u]:
                ei = edge_dir[(u, v)]
                # voltage from u to v
                if petersen_edges[ei][0] == u:
                    vol = voltages[ei]
                else:
                    vol = (-voltages[ei]) % m
                for x in range(m):
                    u_idx = u * m + x
                    v_idx = v * m + ((x + vol) % m)
                    adj[u_idx].append(v_idx)
        return adj

    def find_girth(adj, target_girth):
        """Find the girth using BFS from each vertex. Returns girth or inf if none < target."""
        n = len(adj)
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth < target_girth:
                                return min_girth
        return min_girth

    def find_short_cycle_edges(adj, target_girth):
        """Find a cycle of length < target_girth and return the set of base-edge indices involved."""
        n = len(adj)
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            # Reconstruct cycle and map to base edges
                            # Get path from u to start and v to start
                            path_u = []
                            cur = u
                            while cur != start:
                                path_u.append(cur)
                                cur = parent[cur]
                            path_u.append(start)
                            path_u.reverse()

                            path_v = []
                            cur = v
                            while cur != start:
                                path_v.append(cur)
                                cur = parent[cur]
                            path_v.append(start)
                            path_v.reverse()

                            # Cycle is path_u + reversed path_v[1:]
                            cycle_nodes = path_u + path_v[1:][::-1]
                            # Map to base edges
                            base_edges_used = set()
                            for i in range(len(cycle_nodes) - 1):
                                a = cycle_nodes[i]
                                b = cycle_nodes[i + 1]
                                base_a = a // m
                                base_b = b // m
                                if (base_a, base_b) in edge_dir:
                                    base_edges_used.add(edge_dir[(base_a, base_b)])
                            return base_edges_used
        return None

    # Try primes
    primes = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]

    for m in primes:
        if time.time() > deadline:
            break

        found = False
        best_voltages = None
        best_girth = 0

        for attempt in range(500):
            if time.time() > deadline:
                break

            voltages = [random.randint(1, m - 1) for _ in range(n_edges_base)]

            # Local search
            for step in range(200):
                if time.time() > deadline:
                    break

                adj = build_derived(voltages, m)
                girth = find_girth(adj, g)

                if girth >= g:
                    best_voltages = voltages[:]
                    best_girth = girth
                    found = True
                    break

                if girth > best_girth:
                    best_girth = girth
                    best_voltages = voltages[:]

                # Find a short cycle and modify a voltage
                cycle_edges = find_short_cycle_edges(adj, g)
                if cycle_edges is None or len(cycle_edges) == 0:
                    # Random modification
                    ei = random.randint(0, n_edges_base - 1)
                    voltages[ei] = random.randint(1, m - 1)
                else:
                    # Modify a random edge in the cycle
                    ei = random.choice(list(cycle_edges))
                    voltages[ei] = random.randint(1, m - 1)

            if found:
                break

        if found and best_voltages is not None:
            n_derived = n_base * m
            edges = []
            for ei, (u, v) in enumerate(petersen_edges):
                if petersen_edges[ei][0] == u:
                    vol = best_voltages[ei]
                else:
                    vol = (-best_voltages[ei]) % m
                for x in range(m):
                    u_idx = u * m + x
                    v_idx = v * m + ((x + vol) % m)
                    edges.append((u_idx, v_idx))
            return n_derived, edges

    # Fallback to LUW
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) fallback."""
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
