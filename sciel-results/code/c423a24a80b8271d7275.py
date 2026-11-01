import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # ---- Fallback: The seed LUW graph ----
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

    # Get fallback result immediately to have a valid answer
    fallback_n, fallback_edges = luw_construct(k, g)
    best_n = fallback_n
    best_edges = fallback_edges

    # ---- Fast girth check using BFS with early termination ----
    def check_girth_fast(adj, n, g):
        if n < g:
            return False
        max_depth = (g - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        if du + 1 < max_depth:
                            q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # ---- Petersen graph voltage lift approach ----
    # The Petersen graph has girth 5. A voltage lift of size m has n=10m vertices.
    # Girth of lift >= g iff every closed walk in Petersen of length < g has non-zero voltage sum mod m.
    # We search for voltage assignments (shifts on the 15 edges) that satisfy this.

    def voltage_lift_search(g, max_m, time_budget):
        """Search for a voltage lift of the Petersen graph with girth >= g."""
        # Petersen graph: 10 vertices, 15 edges
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
        ]
        n_base = 10
        E = len(petersen_edges)  # 15

        # Edge index map for quick lookup
        edge_idx = {}
        for i, (u, v) in enumerate(petersen_edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i

        # Adjacency with edge indices
        adj_edges = [[] for _ in range(n_base)]
        for i, (u, v) in enumerate(petersen_edges):
            adj_edges[u].append((v, i))
            adj_edges[v].append((u, i))

        # Precompute all closed walks of length L for L = 5 to g-1
        # Represent each as a tuple of edge indices (the multiset, sorted)
        # A closed walk is a sequence of edges forming a cycle back to start.
        # For voltage sum, only the multiset of edges matters (sum is commutative).
        # But we need to generate all possible multisets that form closed walks.

        # Generate closed walks via DFS
        # To keep it manageable, we generate walks up to length g-1
        # and deduplicate by sorted edge tuple.

        def gen_closed_walks(max_len):
            """Generate all distinct closed walks (by edge multiset) up to max_len."""
            walks = set()
            for start in range(n_base):
                # DFS: (current_vertex, path_edges_tuple, length)
                stack = [(start, (), 0)]
                while stack:
                    curr, path, length = stack.pop()
                    if length > 0 and curr == start:
                        if 5 <= length <= max_len:
                            # Canonical form: sorted edge indices
                            key = tuple(sorted(path))
                            walks.add(key)
                    if length < max_len:
                        for (v, eidx) in adj_edges[curr]:
                            # Avoid immediate backtracking (to reduce redundant walks)
                            # Actually, for voltage sum checking, we need ALL closed walks,
                            # including those that backtrack. But backtracking walks
                            # (e.g., go out and back on same edge) have even length
                            # and their voltage sum is 0 always (v + (-v) = 0).
                            # So they are always "forbidden" - they create 2-cycles in the lift.
                            # Wait, no: in a simple graph, there are no 2-cycles.
                            # A backtrack walk e, e has voltage v_e + (-v_e) = 0, length 2.
                            # But the Petersen graph has no 2-cycles, so a closed walk
                            # of length 2 would be: go along edge e, then back along e.
                            # This is a valid closed walk. Its voltage sum is 0.
                            # This would mean the lift always has a 2-cycle? No!
                            # In the lift, going along edge e from (u, i) to (v, i+v_e),
                            # then back along edge e from (v, i+v_e) to (u, i+v_e+(-v_e)) = (u, i).
                            # This is a closed walk of length 2 in the lift, but it's not
                            # a cycle (it repeats edges). The girth is about simple cycles.
                            # So we should NOT count backtracking walks.
                            # A simple cycle in the lift corresponds to a closed walk
                            # in the base that does not immediately backtrack.
                            # Actually, more precisely: a simple cycle in the lift
                            # projects to a closed walk in the base where no edge is
                            # traversed in both directions consecutively.
                            # Let's just avoid immediate backtracking in the base walk.
                            if path and path[-1] == eidx:
                                continue
                            stack.append((v, path + (eidx,), length + 1))
            return walks

        # Precompute closed walks
        closed_walks = gen_closed_walks(g - 1)
        # Convert to list of edge count vectors for faster sum computation
        # Each walk is a tuple of edge indices. For a given voltage assignment,
        # the sum is sum(voltage[e] for e in walk).
        # We can precompute for each walk the list of edge indices.
        walk_edges = [list(w) for w in closed_walks]
        n_walks = len(walk_edges)

        # Now search for voltage assignments and m such that
        # for all walks, sum(voltage[e] for e in walk) % m != 0

        # Strategy:
        # 1. Fix m (try small m first, since n=10m should be small).
        # 2. Assign voltages to edges one by one, pruning when a walk sum becomes 0 mod m.

        # For efficiency, we'll use a backtracking search with pruning.
        # Order edges so that edges appearing in many short walks are assigned first.

        # Count how many walks use each edge
        edge_freq = [0] * E
        for w in walk_edges:
            for e in w:
                edge_freq[e] += 1

        # Assign edges in order of decreasing frequency (most constrained first)
        edge_order = sorted(range(E), key=lambda e: -edge_freq[e])

        # For each m, try to find a voltage assignment.
        # Voltages can be any integers mod m. We'll try small values.

        # Moore bound: n >= 512, so m >= 52 (since n=10m).
        # But let's try m from 52 upward.

        # For a given m, we need voltages in Z_m such that no closed walk sums to 0.
        # This is a constraint satisfaction problem.

        # Use randomized local search: start with random voltages, then fix violations.
        # Or use backtracking for small m.

        # Let's try a hybrid: for small m, use backtracking. For larger m, use random + local repair.

        def try_backtracking(m, max_nodes=50000):
            """Try to find voltage assignment via backtracking. Returns list of voltages or None."""
            voltages = [None] * E
            walk_sums = [0] * n_walks  # Current sum mod m for each walk
            walk_counts = [0] * n_walks  # How many edges assigned in each walk
            walk_len = [len(w) for w in walk_edges]
            nodes = [0]

            # Precompute for each edge, which walks it belongs to
            edge_to_walks = [[] for _ in range(E)]
            for wi, w in enumerate(walk_edges):
                for e in w:
                    edge_to_walks[e].append(wi)

            def backtrack(pos):
                if nodes[0] > max_nodes:
                    return False
                if pos == E:
                    # Check all walk sums
                    for wi in range(n_walks):
                        if walk_sums[wi] % m == 0:
                            return False
                    return True

                e = edge_order[pos]
                # Try different voltage values
                # To reduce symmetry, fix voltage of first edge to 0 (gauge fixing)
                if pos == 0:
                    candidates = [0]
                else:
                    candidates = list(range(m))
                    random.shuffle(candidates)

                for val in candidates:
                    nodes[0] += 1
                    if nodes[0] > max_nodes:
                        return False

                    voltages[e] = val
                    # Update walk sums
                    affected = edge_to_walks[e]
                    for wi in affected:
                        walk_sums[wi] = (walk_sums[wi] + val) % m
                        walk_counts[wi] += 1

                    # Prune: if any completed walk has sum 0, backtrack
                    valid = True
                    for wi in affected:
                        if walk_counts[wi] == walk_len[wi] and walk_sums[wi] % m == 0:
                            valid = False
                            break

                    if valid and backtrack(pos + 1):
                        return True

                    # Undo
                    for wi in affected:
                        walk_sums[wi] = (walk_sums[wi] - val) % m
                        walk_counts[wi] -= 1
                    voltages[e] = None

                return False

            if backtrack(0):
                return voltages
            return None

        def try_random_local(m, max_iters=2000):
            """Try random voltage assignment + local repair."""
            # Start with random voltages
            voltages = [random.randint(0, m - 1) for _ in range(E)]

            # Count violations
            def count_violations():
                count = 0
                for w in walk_edges:
                    s = sum(voltages[e] for e in w) % m
                    if s == 0:
                        count += 1
                return count

            best_violations = count_violations()
            best_voltages = list(voltages)

            for it in range(max_iters):
                if time.time() > deadline:
                    break
                # Pick a random edge and change its voltage
                e = random.randint(0, E - 1)
                old_val = voltages[e]
                new_val = random.randint(0, m - 1)
                if new_val == old_val:
                    continue
                voltages[e] = new_val

                new_violations = count_violations()
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_voltages = list(voltages)
                    if new_violations == 0:
                        return voltages
                else:
                    # Accept with some probability (simulated annealing)
                    if random.random() < 0.3:
                        pass  # keep the change
                    else:
                        voltages[e] = old_val

            if best_violations == 0:
                return best_voltages
            return None

        # Search over m values
        # n = 10*m, we want n < best_n
        m_max = (best_n - 1) // 10
        m_min = 52  # Moore bound / 10

        # Try m values, starting from small
        m_candidates = list(range(m_min, min(m_max + 1, m_min + 200)))
        # Also add some larger m values
        for extra in [m_min + 100, m_min + 200, m_min + 300, m_min + 400, m_min + 500]:
            if extra <= m_max:
                m_candidates.append(extra)
        m_candidates = sorted(set(m_candidates))

        for m in m_candidates:
            if time.time() > deadline:
                break
            n_lift = 10 * m
            if n_lift >= best_n:
                break

            # Try backtracking first (more reliable for small m)
            voltages = None
            if m <= 200:
                voltages = try_backtracking(m, max_nodes=100000)

            if voltages is None:
                # Try random local search
                for attempt in range(5):
                    if time.time() > deadline:
                        break
                    voltages = try_random_local(m, max_iters=1000)
                    if voltages is not None:
                        break

            if voltages is None:
                continue

            # Build the lift graph
            # Vertex (u, i) for u in 0..9, i in 0..m-1
            # Edge between (u, i) and (v, i + voltage[e]) for each base edge (u,v) with voltage voltage[e]
            # The lift is 3-regular (each vertex has 3 neighbors, same as base).

            lift_edges = []
            for (u, v) in petersen_edges:
                eidx = edge_idx[(u, v)]
                vol = voltages[eidx]
                for i in range(m):
                    j = (i + vol) % m
                    # Vertex index: u * m + i, v * m + j
                    a = u * m + i
                    b = v * m + j
                    lift_edges.append((a, b))

            n_lift = 10 * m
            adj = build_adj(lift_edges, n_lift)

            # Verify girth
            if check_girth_fast(adj, n_lift, g):
                return n_lift, lift_edges

        return None

    # Try voltage lift approach
    if k == 3:
        result = voltage_lift_search(g, max_m=1000, time_budget=deadline - time.time())
        if result is not None:
            n_found, edges_found = result
            if n_found < best_n:
                best_n = n_found
                best_edges = edges_found

    # If voltage lift didn't improve, try configuration model + 2-switches as backup
    if time.time() < deadline - 5:
        def count_short_cycles(adj, n, g):
            if n < g:
                return 1000
            max_depth = (g - 1) // 2
            count = 0
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    du = dist[u]
                    if du >= max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            if du + 1 < max_depth:
                                q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                if start == min(u, v):
                                    count += 1
            return count

        def generate_random_3reg(n):
            if n < 4 or n % 2 != 0:
                return None
            stubs = []
            for v in range(n):
                for _ in range(3):
                    stubs.append(v)
            for attempt in range(100):
                random.shuffle(stubs)
                edges = []
                valid = True
                edge_set = set()
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        valid = False
                        break
                    edge_set.add(key)
                    edges.append((u, v))
                if valid and len(edges) == 3 * n // 2:
                    return edges
            return None

        def try_edge_swap(edges, adj, n):
            if len(edges) < 2:
                return None
            e1_idx = random.randint(0, len(edges) - 1)
            e2_idx = random.randint(0, len(edges) - 1)
            if e1_idx == e2_idx:
                return None
            a, b = edges[e1_idx]
            c, d = edges[e2_idx]
            for new_e1, new_e2 in [((a, c), (b, d)), ((a, d), (b, c))]:
                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                    continue
                key1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                key2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
                if key1 == key2:
                    continue
                old_keys = set()
                old_keys.add((min(a, b), max(a, b)))
                old_keys.add((min(c, d), max(c, d)))
                if key1 in old_keys or key2 in old_keys:
                    continue
                new_edges = list(edges)
                new_edges[e1_idx] = new_e1
                new_edges[e2_idx] = new_e2
                new_adj = [list(nei) for nei in adj]
                new_adj[a].remove(b)
                new_adj[b].remove(a)
                new_adj[c].remove(d)
                new_adj[d].remove(c)
                new_adj[new_e1[0]].append(new_e1[1])
                new_adj[new_e1[1]].append(new_e1[0])
                new_adj[new_e2[0]].append(new_e2[1])
                new_adj[new_e2[1]].append(new_e2[0])
                return new_edges, new_adj
            return None

        # Try a few n values near the Moore bound
        moore_bound = 2 * sum(2**i for i in range(g // 2))
        n_candidates = [n for n in range(moore_bound, min(moore_bound + 50, best_n)) if n % 2 == 0]

        for n_target in n_candidates:
            if time.time() > deadline - 2:
                break
            if n_target >= best_n:
                break

            edges = generate_random_3reg(n_target)
            if edges is None:
                continue
            adj = build_adj(edges, n_target)

            if check_girth_fast(adj, n_target, g):
                best_n = n_target
                best_edges = edges
                break

            # Simulated annealing with 2-switches
            current_count = count_short_cycles(adj, n_target, g)
            best_count = current_count
            best_edges_sa = list(edges)
            temperature = 50.0
            for it in range(3000):
                if time.time() > deadline - 2:
                    break
                result = try_edge_swap(edges, adj, n_target)
                if result is None:
                    continue
                new_edges, new_adj = result
                new_count = count_short_cycles(new_adj, n_target, g)
                delta = new_count - current_count
                if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                    edges = new_edges
                    adj = new_adj
                    current_count = new_count
                    if new_count < best_count:
                        best_count = new_count
                        best_edges_sa = list(edges)
                    if new_count == 0:
                        if check_girth_fast(adj, n_target, g):
                            best_n = n_target
                            best_edges = edges
                            break
                temperature *= 0.999

            if best_count == 0 and best_n > n_target:
                if check_girth_fast(build_adj(best_edges_sa, n_target), n_target, g):
                    best_n = n_target
                    best_edges = best_edges_sa

    return best_n, best_edges

# EVOLVE-BLOCK-END
