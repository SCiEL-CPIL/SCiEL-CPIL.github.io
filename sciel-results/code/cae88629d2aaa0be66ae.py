"""Cage graph constructor: 3-regular, girth >= g, minimize n."""
import time
import random

_START = time.time()
_DEADLINE = _START + 140  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _deadline_hit():
    return _elapsed() > _DEADLINE


def construct(k, g):
    """Construct k-regular graph with girth >= g on as few vertices as possible."""
    # For k=3, g=16: bipartite, radius r = g//2 = 7
    # Moore bound lower bound on n: 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1))
    # For k=3, r=7: 2*(1 + 2 + 4 + 8 + 16 + 32 + 64) = 2*127 = 254
    # But actual cage n(3,16) is known to be 936.
    #
    # Strategy: voltage graph on a base graph. Use a cycle of length g as base,
    # with voltages in Z_m. The cover has n = m * |base| vertices.
    # For a cycle C_g with voltage assignments, the cover is a k-regular graph.
    #
    # Better: use a tree-like base (a "cage tree") with voltage assignments
    # to make the cover have girth >= g.
    #
    # Most practical approach for k=3, g=16:
    # Use the known structure: a bipartite graph where one side has "root" vertices
    # and the other has "leaf" vertices, connected via voltage assignments.
    #
    # Practical approach: random 3-regular bipartite graph with girth verification,
    # using local repair to increase girth.
    #
    # For n=936 (known optimum), we can try to construct it directly.
    # But let's use a voltage graph approach that's more systematic.

    if k == 3 and g == 16:
        # Use voltage graph on a 3-regular base with girth >= 16
        # Base: use a cycle C_16 with 3 parallel "edges" via voltage assignment
        # Actually, for 3-regular: use a base graph that's a 3-regular graph
        # with girth >= 16, then lift with voltages.
        #
        # Simplest: construct a 3-regular bipartite graph directly using
        # voltage assignments on a small base.
        #
        # Base graph: a single vertex with 3 self-loops (not simple).
        # Better: use a path/cycle as base.
        #
        # Let me use the standard construction:
        # A (d, g) cage via voltage graph on a tree of depth g/2 - 1.
        # For k=3, g=16: tree of depth 6 (radius 7, so depth 6 from root).
        # The tree has 1 + 2 + 4 + 8 + 16 + 32 + 64 = 127 nodes.
        # But we need to close it into a 3-regular graph, which requires
        # identifying leaves with voltages.
        #
        # Practical: use random 3-regular bipartite graph and verify girth.
        # Start with n = 936 (known optimum) and try to construct.

        n = 936
        half = n // 2  # 468 vertices on each side

        # Construct a 3-regular bipartite graph using voltage assignment
        # on a base cycle. Use a cycle of length L with voltages in Z_m.
        # Total vertices = L * m.
        # For 3-regular: base must be 3-regular.
        #
        # Use base = cycle C_3 (triangle) with 3 edges, lifted by Z_m.
        # But C_3 has girth 3, too small.
        #
        # Use base = C_16 (16-cycle), 2-regular. Add a perfect matching
        # to make it 3-regular. Base has 16 vertices, 24 edges.
        # Lift by Z_m: total = 16*m vertices.
        # For 936: 16*m = 936 -> m = 58.5, not integer.
        #
        # Use base = C_8 with a perfect matching (3-regular on 8 vertices).
        # Lift by Z_m: total = 8*m. For 936: m = 117.
        # Girth of base C_8 with matching: the matching creates 4-cycles,
        # so base girth = 4. Too small.
        #
        # Use base = C_16 with a perfect matching (3-regular on 16 vertices).
        # Base girth: C_16 has girth 16, matching might create shorter cycles.
        # If matching connects opposite vertices (i to i+8), no short cycles.
        # Base girth = 16. Good.
        # Lift by Z_m: total = 16*m. For 936: m = 58.5. Not integer.
        #
        # Use base = C_12 with matching (3-regular on 12 vertices).
        # Matching i to i+6: girth = 12. Lift by Z_m: total = 12*m.
        # For 936: m = 78.
        #
        # Voltage assignment: for each edge in base, assign a voltage in Z_m.
        # The cover graph has girth >= g iff:
        # 1. Base girth >= g (satisfied: 12 < 16, NOT satisfied!)
        #
        # Problem: base girth must be >= g for the cover to have girth >= g.
        # So base must have girth >= 16. Smallest 3-regular graph with girth >= 16
        # is the cage itself (936 vertices). Circular.
        #
        # Alternative: use a base with girth < 16 but voltage assignments
        # that "break" short cycles. This is the voltage graph approach.
        #
        # For a base cycle C_L with voltages v_1, ..., v_L in Z_m:
        # The cover is a 2-regular graph (union of cycles).
        # Girth of cover = L * m / gcd(L, sum(v_i)) if all voltages equal.
        # To get girth >= 16: need L*m/gcd >= 16.
        #
        # For 3-regular: add a matching with voltages.
        #
        # Let me just use a direct random construction with girth checking.

        # Random 3-regular bipartite graph with n = 936
        # Use configuration model
        random.seed(42)

        def make_3reg_bipartite(n):
            half = n // 2
            # Left side: 0..half-1, Right side: half..n-1
            # Each vertex has degree 3
            left_stubs = []
            right_stubs = []
            for i in range(half):
                left_stubs.extend([i] * 3)
                right_stubs.extend([i + half] * 3)

            random.shuffle(left_stubs)
            random.shuffle(right_stubs)

            edges = []
            edge_set = set()
            valid = True
            for l, r in zip(left_stubs, right_stubs):
                if l == r:
                    valid = False
                    break
                key = (min(l, r), max(l, r))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append(key)

            if not valid:
                return None, None
            return n, edges

        def check_girth(n, edges, min_girth):
            """Check if graph has girth >= min_girth using BFS from each vertex."""
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)

            # BFS from each vertex, track shortest cycle
            # For bipartite graph, cycles are even length
            # Use non-backtracking BFS
            for start in range(n):
                if _deadline_hit():
                    return False
                # BFS tracking (vertex, parent, depth)
                visited = {start: (-1, 0)}  # vertex -> (parent, depth)
                queue = [(start, -1, 0)]
                found_cycle = False
                while queue:
                    if _deadline_hit():
                        return False
                    node, parent, depth = queue.pop(0)
                    if depth >= min_girth // 2:
                        continue
                    for nb in adj[node]:
                        if nb == parent:
                            continue
                        if nb in visited:
                            # Found a cycle
                            cycle_len = depth + visited[nb][1] + 1
                            if cycle_len < min_girth:
                                return False
                        else:
                            visited[nb] = (node, depth + 1)
                            queue.append((nb, node, depth + 1))
            return True

        # Try to construct with n = 936
        best_n = None
        best_edges = None

        for attempt in range(100):
            if _deadline_hit():
                break
            n, edges = make_3reg_bipartite(936)
            if n is None:
                continue
            if check_girth(n, edges, 16):
                best_n = n
                best_edges = edges
                break

        if best_n is not None:
            return best_n, best_edges

        # Fallback: return a valid (but larger) graph if we can't find girth 16
        # Use the Lazebnik-Ustimenko-Woldar construction from the seed
        q = 3
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # For g=16: d = 11 (since 11 is odd)
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
                    y[c] = (x[c] + y[a] * x[b]) % q if side == 0 else (x[c] - x[a] * y[b]) % q
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

    # Generic fallback for other k, g
    # Use the seed construction
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
