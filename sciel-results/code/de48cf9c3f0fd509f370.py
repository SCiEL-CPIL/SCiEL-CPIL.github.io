import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before the 150 s kill

    def luw_construct(k, g, d_override=None, rule_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
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
        if rule_override is not None:
            rule = rule_override
        else:
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

    def verify_girth(n, edges, target_girth):
        """Certify girth >= target_girth using non-backtracking walks from
        every vertex (as required by the task contract). Also enforces the
        simple, k-regular invariants."""
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or v < 0 or u >= n or v >= n:
                return False
            key = (u, v) if u < v else (v, u)
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        half = (target_girth - 1) // 2
        for s in range(n):
            dist = {}
            queue = deque()
            for nb in adj[s]:
                st = (nb, s)
                dist[st] = 1
                queue.append(st)
            while queue:
                v, p = queue.popleft()
                dcur = dist[(v, p)]
                if dcur >= half:
                    continue
                for w in adj[v]:
                    if w == p:
                        continue
                    st = (w, v)
                    if st in dist:
                        cyc = dcur + dist[st] + 1
                        if cyc < target_girth:
                            return False
                    else:
                        dist[st] = dcur + 1
                        queue.append(st)
        return True

    def fast_girth_check(n, edges, target_girth, check_vertices=3):
        """Quick girth check from a few vertices only. If any short cycle is found,
        reject immediately. If no short cycle found from these vertices, the graph
        might still be valid (false negative possible but unlikely for bad rules)."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        half = (target_girth - 1) // 2
        # Check from vertex 0 and a couple others
        for s in range(min(check_vertices, n)):
            dist = {}
            queue = deque()
            for nb in adj[s]:
                st = (nb, s)
                dist[st] = 1
                queue.append(st)
            while queue:
                v, p = queue.popleft()
                dcur = dist[(v, p)]
                if dcur >= half:
                    continue
                for w in adj[v]:
                    if w == p:
                        continue
                    st = (w, v)
                    if st in dist:
                        cyc = dcur + dist[st] + 1
                        if cyc < target_girth:
                            return False
                    else:
                        dist[st] = dcur + 1
                        queue.append(st)
        return True

    def gen_random_rule(d, rng):
        """Generate a random rule parameterization."""
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            # Random choice of (a, b) indices, biased toward nearby indices
            # which tend to produce better girth properties
            r = rng.random()
            if r < 0.3:
                a, b = 1, rng.randint(max(1, c - 4), c - 1)
            elif r < 0.6:
                a, b = rng.randint(max(1, c - 4), c - 1), 1
            elif r < 0.8:
                a, b = c - 1, c - 2
            else:
                a, b = rng.randint(1, c - 1), rng.randint(1, c - 1)
            rule[c] = (a, b)
        return rule

    def gen_structured_rule(d, variant, rng):
        """Generate structured rule variants with some randomness."""
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            if variant == 0:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            elif variant == 1:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 1), (c0, 1), (c0 + 1, 1), (1, c0)][r]
            elif variant == 2:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(2, c0 - 1), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            elif variant == 3:
                # Shifted pattern
                r, c0 = (c - 5 + variant) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            elif variant == 4:
                # Alternating
                if c % 2 == 1:
                    rule[c] = (1, c - 2)
                else:
                    rule[c] = (c - 1, 1)
            elif variant == 5:
                # Every 3 pattern
                if c % 3 == 0:
                    rule[c] = (1, c - 2)
                elif c % 3 == 1:
                    rule[c] = (c - 1, 1)
                else:
                    rule[c] = (1, c - 1)
            elif variant == 6:
                # Randomized structured
                base = [(1, c - 2), (c - 1, 1), (1, c - 1), (c - 2, 1)]
                rule[c] = base[rng.randint(0, 3)]
            else:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    if k != 3 or g != 16:
        return luw_construct(k, g)

    best_n = float('inf')
    best_edges = None
    rng = random.Random(42)

    # First try the known-good default construction
    n, edges = luw_construct(k, g)
    if verify_girth(n, edges, g):
        best_n = n
        best_edges = edges

    # Search over different d values and rule variants
    # Use fast girth check to quickly filter, then full verify for candidates
    for d in [7, 8, 9, 10, 11, 12, 13, 14]:
        if time.time() > deadline - 5:
            break
        # Try structured variants first
        for variant in range(8):
            if time.time() > deadline - 3:
                break
            rule = gen_structured_rule(d, variant, rng)
            n, edges = luw_construct(k, g, d_override=d, rule_override=rule)
            if n >= best_n:
                continue
            # Quick check before expensive full verification
            if fast_girth_check(n, edges, g, check_vertices=2):
                if verify_girth(n, edges, g):
                    best_n = n
                    best_edges = edges

        # Try random rules
        num_random = 20 if d <= 9 else 10
        for _ in range(num_random):
            if time.time() > deadline - 3:
                break
            rule = gen_random_rule(d, rng)
            n, edges = luw_construct(k, g, d_override=d, rule_override=rule)
            if n >= best_n:
                continue
            if fast_girth_check(n, edges, g, check_vertices=2):
                if verify_girth(n, edges, g):
                    best_n = n
                    best_edges = edges

    if best_edges is not None:
        return best_n, best_edges

    return luw_construct(k, g)
