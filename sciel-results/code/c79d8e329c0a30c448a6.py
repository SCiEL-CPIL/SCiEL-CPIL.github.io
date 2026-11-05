import time
import random
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    # For k=3, g=16: Use voltage assignment over Petersen graph
    # with fast cycle-sum verification
    if k == 3:
        # Petersen graph edges (10 vertices, 15 edges, girth 5)
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner pentagram
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
        ]
        n_base = 10
        n_edges = len(petersen_edges)

        # Edge index lookup
        edge_idx = {}
        for i, (u, v) in enumerate(petersen_edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i

        # Build base adjacency
        base_adj = [[] for _ in range(n_base)]
        for (u, v) in petersen_edges:
            base_adj[u].append(v)
            base_adj[v].append(u)

        # Enumerate all simple cycles of length < g in Petersen graph
        # Petersen has girth 5, so we check cycles of length 5, 6, 7, 8, 9, 10
        def find_all_cycles(adj, max_len):
            """Find all simple cycles up to max_len in a small graph."""
            cycles = []
            n = len(adj)
            # Fix start vertex to 0 to avoid duplicates, but we need all cycles
            # Actually, for voltage checking, we need each distinct cycle as a set of edge indices
            # Let's use a canonical form: start from smallest vertex, go to smallest neighbor
            seen = set()
            for start in range(n):
                for first in adj[start]:
                    if first < start:
                        continue
                    # DFS to find cycles
                    stack = [(start, first, [first], [(start, first)])]
                    while stack:
                        node, prev, path, edge_path = stack.pop()
                        if len(path) >= max_len:
                            continue
                        if node == start and len(path) >= 3:
                            # Found a cycle
                            # Canonical form: sort edges
                            edge_set = frozenset(edge_path)
                            if edge_set not in seen:
                                seen.add(edge_set)
                                cycles.append(edge_path)
                            continue
                        for nb in adj[node]:
                            if nb == prev:
                                continue
                            if nb == start and len(path) + 1 >= 3:
                                # Close the cycle
                                new_edge_path = edge_path + [(node, nb)]
                                edge_set = frozenset(new_edge_path)
                                if edge_set not in seen:
                                    seen.add(edge_set)
                                    cycles.append(new_edge_path)
                            elif nb > start:  # Avoid duplicates
                                stack.append((nb, node, path + [nb], edge_path + [(node, nb)]))
            return cycles

        # For voltage checking, we need cycles as sequences of edge indices (with signs)
        # Actually, for a cycle in the lift to close, the sum of signed voltages must be 0 mod m
        # Let's pre-compute the signed edge sequences for each cycle
        # A cycle v0 -> v1 -> ... -> v_{L-1} -> v0 has edges e0, e1, ..., e_{L-1}
        # The voltage sum is sum of voltages along the cycle (each edge counted once)
        # But the sign depends on direction. For a simple cycle, each edge is traversed once.
        # The net voltage is the sum of voltages of edges in the cycle (mod m)
        # Wait, that's not quite right. In a voltage graph, the voltage of a path is the sum of voltages
        # of the edges in the path (with appropriate signs based on orientation).
        # For an undirected graph, we orient each edge. The voltage of edge (u,v) is a value.
        # When traversing from u to v, we add the voltage; from v to u, we subtract.
        # So for a cycle, we need to sum with signs.

        # Let me re-think. In a voltage assignment on an undirected graph:
        # Each edge has a voltage value. When we traverse an edge in one direction, we add the voltage;
        # in the opposite direction, we subtract.
        # For a cycle to be "closed" in the lift (i.e., a cycle in the lifted graph),
        # the sum of signed voltages around the cycle must be 0 mod m.

        # Pre-compute signed edge index sequences for all cycles of length < g
        # in the Petersen graph
        def get_signed_cycles(adj, n_base, g):
            """Get all cycles of length < g as lists of (edge_idx, sign) tuples."""
            cycles = []
            seen = set()
            for start in range(n_base):
                for first in adj[start]:
                    if first < start:
                        continue
                    stack = [(first, start, [(start, first)])]
                    while stack:
                        node, prev, edge_seq = stack.pop()
                        if len(edge_seq) >= g:
                            continue
                        if node == start and len(edge_seq) >= 3:
                            # Found a cycle
                            # Canonical: use frozenset of (edge_idx, 1) pairs
                            # Each edge appears once with sign +1 in this direction
                            key = tuple(sorted((edge_idx[(edge_seq[i][0], edge_seq[i][1])], 1) 
                                             for i in range(len(edge_seq))))
                            if key not in seen:
                                seen.add(key)
                                # Store as list of (edge_idx, sign)
                                signed = [(edge_idx[(edge_seq[i][0], edge_seq[i][1])], 1) 
                                          for i in range(len(edge_seq))]
                                cycles.append(signed)
                            continue
                        for nb in adj[node]:
                            if nb == prev:
                                continue
                            if nb == start and len(edge_seq) + 1 >= 3:
                                new_seq = edge_seq + [(node, nb)]
                                key = tuple(sorted((edge_idx[(new_seq[i][0], new_seq[i][1])], 1) 
                                                 for i in range(len(new_seq))))
                                if key not in seen:
                                    seen.add(key)
                                    signed = [(edge_idx[(new_seq[i][0], new_seq[i][1])], 1) 
                                              for i in range(len(new_seq))]
                                    cycles.append(signed)
                            elif nb > start:
                                stack.append((nb, node, edge_seq + [(node, nb)]))
            return cycles

        signed_cycles = get_signed_cycles(base_adj, n_base, g)

        # Fast girth check: for a voltage assignment, check if any cycle has sum 0 mod m
        def fast_girth_check(voltages, m, signed_cycles):
            for cycle in signed_cycles:
                total = 0
                for eidx, sign in cycle:
                    total += sign * voltages[eidx]
                if total % m == 0:
                    return False
            return True

        # Build the lifted graph
        def build_lifted_graph(voltages, m):
            n_total = n_base * m
            adj = [[] for _ in range(n_total)]
            edges = []
            for (u, v) in petersen_edges:
                eidx = edge_idx[(u, v)]
                vol = voltages[eidx]
                for t in range(m):
                    u_idx = u * m + t
                    v_idx = v * m + ((t + vol) % m)
                    adj[u_idx].append(v_idx)
                    adj[v_idx].append(u_idx)
                    edges.append((u_idx, v_idx))
            return n_total, adj, edges

        # Full BFS girth verification
        def verify_girth(adj, n, g):
            for s in range(n):
                visited = set()
                queue = [(s, -1)]
                visited.add((s, -1))
                depth = 0
                while queue and depth < g - 1:
                    new_queue = []
                    for (node, prev) in queue:
                        for nb in adj[node]:
                            if nb == prev:
                                continue
                            if nb == s and depth + 1 >= 3:
                                return False
                            if (nb, node) not in visited:
                                visited.add((nb, node))
                                new_queue.append((nb, node))
                    queue = new_queue
                    depth += 1
            return True

        # Search for voltage assignment
        best_result = None
        best_n = float('inf')

        # Try various m values
        for m in [51, 52, 53, 54, 55, 56, 57, 58, 60, 65, 70, 80, 90, 100]:
            if time_left() < 10:
                break

            n_total = n_base * m
            if n_total >= best_n:
                continue

            attempts = 0
            max_attempts = 200000

            while attempts < max_attempts and time_left() > 15:
                attempts += 1
                voltages = [random.randint(0, m-1) for _ in range(n_edges)]

                if fast_girth_check(voltages, m, signed_cycles):
                    n_t, adj, edges = build_lifted_graph(voltages, m)
                    if verify_girth(adj, n_t, g):
                        # Verify simple graph and degree
                        edge_set = set()
                        valid = True
                        for (u, v) in edges:
                            if u == v:
                                valid = False
                                break
                            key = (min(u, v), max(u, v))
                            if key in edge_set:
                                valid = False
                                break
                            edge_set.add(key)

                        if valid:
                            deg = [0] * n_t
                            for (u, v) in edges:
                                deg[u] += 1
                                deg[v] += 1
                            if all(d == k for d in deg):
                                best_n = n_t
                                best_result = (n_t, edges)
                                break

            if best_result is not None:
                break

        if best_result is not None:
            return best_result

    # Fallback: LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
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

        n = len(order)
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        # Verify girth
        min_cycle = float('inf')
        for s in range(n):
            dist = {s: 0}
            parent = {s: -1}
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if nb in dist:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            break
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                if min_cycle < g:
                    break
            if min_cycle < g:
                break

        return n, edges

    if k != 3:
        return luw_construct(k, g)

    # Fallback to LUW for k=3 if voltage search failed
    return luw_construct(k, g)
