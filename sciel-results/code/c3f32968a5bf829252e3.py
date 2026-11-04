import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (fast, early exit) ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- LUW construction ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def generate_rules(d, variant=0, offset=0):
        base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                base[c] = (1, c - 2) if (c + offset) % 2 == 0 else (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if (c + offset) % 3 == 0:
                    base[c] = (1, c - 3)
                elif (c + offset) % 3 == 1:
                    base[c] = (c - 1, 1)
                else:
                    base[c] = (c - 2, 1)
        elif variant == 3:
            for c in range(5, d + 1):
                base[c] = (1, c - (1 + (c + offset) % 2))
        elif variant == 4:
            for c in range(5, d + 1):
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][(c + offset) % 4]
        elif variant == 5:
            for c in range(5, d + 1):
                if (c + offset) % 2 == 0:
                    base[c] = (1, c - 2)
                else:
                    base[c] = (c - 3, 1)
        elif variant == 6:
            for c in range(5, d + 1):
                idx = (c * 3 + offset) % 4
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][idx]
        else:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return base

    def build_luw(q, d, rule, time_limit):
        add, sub, mul = _luw_ops(q)

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
            if time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best = None
        best_n = float('inf')

        for d_try in range(d, d + 8):
            if remaining() < 15:
                break
            for variant in range(7):
                for offset in range(5):
                    if remaining() < 12:
                        break
                    rule = generate_rules(d_try, variant, offset)
                    res = build_luw(q, d_try, rule, deadline - 12)
                    if res is None:
                        continue
                    n, edges = res
                    if n >= best_n:
                        continue
                    gc = check_girth(n, edges, g, deadline - 8)
                    if gc == -1:
                        continue
                    if gc >= g:
                        best, best_n = (n, edges), n
                        if n < 10000:
                            break
                if best is not None and best_n < 10000:
                    break
            if best is not None and best_n < 10000:
                break

        if best is None:
            rule = generate_rules(d, 0, 0)
            res = build_luw(q, d, rule, deadline - 2)
            if res is not None:
                best = res
        return best

    # ---------------- Ramanujan cubic graphs (k=3) ----------------
    def build_ramanujan(q):
        def mul(a, b):
            return ((a[0]*b[0] + a[1]*b[2]) % q, (a[0]*b[1] + a[1]*b[3]) % q,
                    (a[2]*b[0] + a[3]*b[2]) % q, (a[2]*b[1] + a[3]*b[3]) % q)

        def det(a):
            return (a[0]*a[3] - a[1]*a[2]) % q

        def inv(a):
            d = det(a)
            invd = pow(d, q - 2, q)
            return ((a[3]*invd) % q, (-a[1]*invd) % q, (-a[2]*invd) % q, (a[0]*invd) % q)

        def norm(a):
            return (a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]) % q

        def involution(a):
            if det(a) != 1:
                return False
            m = mul(a, a)
            n = norm(m)
            return n == 0 or n == 2

        I = (1, 0, 0, 1)
        A = (0, 1, q - 1, 0)
        B = (0, 1, 1, 0)
        if not (involution(A) and involution(B)):
            return None
        C = None
        for s in range(1, q):
            cand = (1, 0, s, 1)
            if involution(cand):
                C = cand
                break
        if C is None:
            return None
        G = [I, A, B, C,
             mul(A, B), mul(B, A), mul(A, C), mul(C, A),
             mul(B, C), mul(C, B), mul(A, B), mul(B, A)]
        seen, gens = set(), []
        for m in G:
            key = m if (m[0], m[1], m[2], m[3]) <= (q - m[0] % q, q - m[1] % q, q - m[2] % q, q - m[3] % q) else (q - m[0] % q, q - m[1] % q, q - m[2] % q, q - m[3] % q)
            if key not in seen:
                seen.add(key)
                gens.append(m)
        if len(gens) != 3:
            return None
        start = I
        index = {start: 0}
        order = [start]
        edges = []
        for i in range(len(order)):
            v = order[i]
            vi = inv(v)
            for w in gens:
                u = mul(vi, w)
                if u not in index:
                    index[u] = len(order)
                    order.append(u)
                if index[v] < index[u]:
                    edges.append((index[v], index[u]))
        return len(order), edges

    # ---------------- cage completion: build NB tree, merge boundary via CSP ----------------
    def cage_complete(k, g, time_limit):
        # The non-backtracking tree of depth h has boundary vertices each needing
        # (k-1) edges to complete k-regularity. We build the tree, then solve a
        # matching/CSP over the boundary to close the graph without short cycles.
        if g % 2 != 0:
            return None
        h = g // 2
        # Number of boundary vertices in a k-regular NB tree of depth h:
        # depth 0: 1, depth i (i>=1): (k-1)^(i-1)*k ... actually NB tree:
        # root has k children; each subsequent node has (k-1) children.
        # depth i has k*(k-1)^(i-1) nodes for i>=1.
        # boundary = depth h nodes = k*(k-1)^(h-1)
        boundary_count = k * (k - 1) ** (h - 1)
        # Each boundary vertex needs (k-1) more edges. Total half-edges = boundary_count*(k-1).
        # This is a perfect matching problem on boundary_count vertices with (k-1) edges each
        # -> a (k-1)-regular graph on boundary_count vertices.
        if boundary_count < 2:
            return None
        # For small boundary_count we can do a deterministic random-ish (k-1)-regular
        # graph and check girth; try a few seeds.
        import random
        best = None
        best_n = float('inf')
        for seed in range(30):
            if time.time() > time_limit:
                break
            rng = random.Random(seed)
            # Build the tree explicitly.
            # vertices: 0 = root.
            # We'll generate children in order.
            # depth 0: [0]
            # For each node at depth < h, add children.
            # root (depth0) has k children; others have (k-1) children.
            # We track depth.
            verts = [0]
            depth = [0]
            parent = [-1]
            edges = []
            # children lists
            children = [[]]
            # BFS
            idx = 0
            while idx < len(verts):
                d = depth[idx]
                if d >= h:
                    idx += 1
                    continue
                num_children = k if d == 0 else (k - 1)
                for _ in range(num_children):
                    nv = len(verts)
                    verts.append(nv)
                    depth.append(d + 1)
                    parent.append(idx)
                    children.append([])
                    children[idx].append(nv)
                    edges.append((idx, nv))
                idx += 1
            n_tree = len(verts)
            # boundary = vertices at depth h
            boundary = [v for v in range(n_tree) if depth[v] == h]
            # Need to add (k-1) edges among boundary to make k-regular.
            # Build a (k-1)-regular graph on boundary.
            m = len(boundary)
            # We must avoid creating cycles shorter than g.
            # Simple approach: try to build (k-1)-regular graph and check.
            # For (k-1) >= 2, we need a 2-regular+ graph.
            # Use a random regular graph construction.
            deg = k - 1
            if deg < 1:
                return None
            # Try to construct deg-regular graph on m vertices via configuration model,
            # then dedupe; if duplicates, retry.
            ok = False
            for attempt in range(200):
                if time.time() > time_limit:
                    break
                # half-edges
                half = list(range(m)) * deg
                rng.shuffle(half)
                pairs = []
                valid = True
                seen_pairs = set()
                for i in range(0, len(half), 2):
                    a, b = half[i], half[i + 1]
                    if a == b:
                        valid = False
                        break
                    key = (min(a, b), max(a, b))
                    if key in seen_pairs:
                        valid = False
                        break
                    seen_pairs.add(key)
                    pairs.append(key)
                if not valid:
                    continue
                # Build full edge list
                all_edges = list(edges)
                bidx = {v: i for i, v in enumerate(boundary)}
                for a, b in pairs:
                    all_edges.append((boundary[a], boundary[b]))
                n = n_tree
                # check simple: ensure no duplicates overall
                eset = set()
                dup = False
                for a, b in all_edges:
                    key = (min(a, b), max(a, b))
                    if key in eset:
                        dup = True
                        break
                    eset.add(key)
                if dup:
                    continue
                # check k-regular
                degs = [0] * n
                for a, b in all_edges:
                    degs[a] += 1
                    degs[b] += 1
                if any(d != k for d in degs):
                    continue
                # check girth
                gc = check_girth(n, all_edges, g, time_limit)
                if gc == -1:
                    break
                if gc >= g:
                    ok = True
                    if n < best_n:
                        best, best_n = (n, all_edges), n
                    break
            if best is not None and best_n <= 2000:
                break
        return best

    # ---------------- main ----------------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    # k == 3: try cage completion first (may give smaller graphs), then Ramanujan, then LUW.
    best = None
    best_n = float('inf')

    # Try cage completion with a budget
    if remaining() > 25:
        cc = cage_complete(k, g, deadline - 20)
        if cc is not None:
            best, best_n = cc

    # Try Ramanujan graphs (smaller) to improve
    for q in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]:
        if remaining() < 8:
            break
        res = build_ramanujan(q)
        if res is None:
            continue
        n, edges = res
        if n >= best_n:
            continue
        gc = check_girth(n, edges, g, deadline - 5)
        if gc == -1:
            continue
        if gc >= g:
            best, best_n = (n, edges), n

    if best is not None:
        return best

    # fall back to LUW (guaranteed valid)
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
