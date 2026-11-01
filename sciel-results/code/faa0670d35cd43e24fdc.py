"""Stronger program: build the full Lazebnik-Ustimenko-Woldar CD(d,q) graph."""
import time
import itertools

def _time_left(deadline):
    return deadline - time.time()

def _luw_graph(k, d):
    """Build the full Lazebnik-Ustimenko-Woldar CD(d, k) graph.
    Returns (n, edges). Bipartite with 2*q^d vertices.
    Girth is d+5 (d odd) or d+6 (d even)."""
    q = k
    if q == 4:
        mult_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
        ]
        def add(a, b): return a ^ b
        def sub(a, b): return a ^ b
        def mul(a, b): return mult_table[a][b]
    else:
        def add(a, b): return (a + b) % q
        def sub(a, b): return (a - b) % q
        def mul(a, b): return (a * b) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    vectors = list(itertools.product(range(q), repeat=d))
    vec_idx = {v: i for i, v in enumerate(vectors)}
    half = q ** d
    n = 2 * half
    edges = []
    for pi, p in enumerate(vectors):
        for l1 in range(q):
            l = [0] * d
            l[0] = l1
            for c in range(2, d + 1):
                a, b = rule[c]
                lc = add(p[c - 1], mul(l[a - 1], p[b - 1]))
                l[c - 1] = lc
            li = vec_idx[tuple(l)]
            edges.append((pi, half + li))
    return n, edges

def construct(k, g):
    deadline = time.time() + 140.0
    best = None

    for d in range(2, 30):
        if _time_left(deadline) < 5.0:
            break
        expected_girth = d + 5 if d % 2 == 1 else d + 6
        if expected_girth < g:
            continue
        if expected_girth > g + 10:
            break
        try:
            n, edges = _luw_graph(k, d)
        except Exception:
            continue
        # Quick sanity: check regularity
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(x != k for x in deg):
            continue
        if best is None or n < best[0]:
            best = (n, edges)

    if best is not None:
        return best
    raise RuntimeError("No valid graph found")
