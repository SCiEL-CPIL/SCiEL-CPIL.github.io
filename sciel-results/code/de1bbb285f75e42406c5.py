import time
from collections import deque

_T0 = time.time()
_DEADLINE = _T0 + 135.0

# Petersen graph: 10 vertices, 3-regular, girth 5
_PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer 5-cycle
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # inner 5-cycle
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),  # spokes
]
_PETERSEN_ADJ = [[] for _ in range(10)]
for u, v in _PETERSEN_EDGES:
    _PETERSEN_ADJ[u].append(v)
    _PETERSEN_ADJ[v].append(u)


def _thue_morse(n):
    """Return the first n bits of the Thue-Morse word (0-indexed)."""
    bits = []
    for i in range(n):
        bits.append(bin(i).count('1') & 1)
    return bits


def _build_lift(m, voltages):
    """Build the voltage lift of the Petersen graph over Z_m.
    voltages: list of 15 ints, one per edge in _PETERSEN_EDGES.
    Returns (n, edges) for the lifted 3-regular graph on 10*m vertices."""
    n = 10 * m
    edges = []
    for idx, (u, v) in enumerate(_PETERSEN_EDGES):
        w = voltages[idx] % m
        for i in range(m):
            a = u * m + i
            b = v * m + ((i + w) % m)
            edges.append((a, b))
    return n, edges


def _verify_girth_edges(n, edges, g):
    """Build adjacency list from edges and verify girth >= g via non-backtracking BFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    max_depth = (g - 1) // 2
    for start in range(n):
        # visited[node] = (depth, parent)
        visited = {start: (0, -1)}
        queue = deque([(start, 0, -1)])
        while queue:
            node, depth, parent = queue.popleft()
            if depth >= max_depth:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb in visited:
                    cycle_len = depth + 1 + visited[nb][0]
                    if cycle_len < g:
                        return False
                else:
                    visited[nb] = (depth + 1, node)
                    queue.append((nb, depth + 1, node))
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
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
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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


def _find_thue_morse_lift(k, g, deadline):
    """Try voltage lifts of Petersen over Z_m with Thue-Morse-derived voltages.
    Returns (n, edges) if a valid girth>=g graph is found, else None."""
    if k != 3:
        return None
    # Thue-Morse bits for voltage assignment
    tm = _thue_morse(15)
    # We want voltages in Z_m that are "spread out". Use tm bits to pick from a
    # set of candidate multipliers, then scale.
    # Strategy: for each m, assign voltages as tm[i] * c_i mod m where c_i are
    # distinct-ish constants to maximize dispersion.
    # Use a deterministic set of multipliers derived from small primes / quadratic residues.
    # To get high dispersion, use voltages[i] = tm[i] * s_i mod m where s_i are
    # chosen so that the 15 values (with signs from walk structure) avoid small sums.
    # Simpler: voltages[i] = (tm[i] + 1) * (i + 1) * R mod m, where R is a
    # primitive-root-like constant. But let's just use a family and test.
    
    # Candidate multiplier sets (15 values each), to be reduced mod m.
    # Set A: based on Thue-Morse weighted by index
    mult_A = [(tm[i] + 1) * (i + 1) for i in range(15)]
    # Set B: Thue-Morse bits mapped to a spread set
    spread = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    mult_B = [spread[i] * (2 * tm[i] - 1) for i in range(15)]
    # Set C: mix
    mult_C = [(tm[i] + 1) * spread[i] for i in range(15)]

    mult_sets = [mult_A, mult_B, mult_C]

    # m ranges: n = 10*m. Moore bound 512 -> m >= 52. Known best 936 -> m <= 94.
    # Try small m first (smaller n is better).
    for m in range(52, 95):
        if time.time() > deadline - 5:
            return None
        for mults in mult_sets:
            voltages = [mults[i] % m for i in range(15)]
            n, edges = _build_lift(m, voltages)
            if _verify_girth_edges(n, edges, g):
                return n, edges
        # Also try a few deterministic perturbations of mult_A
        for shift in range(1, 6):
            if time.time() > deadline - 5:
                return None
            voltages = [(mult_A[i] + shift * (i + 1)) % m for i in range(15)]
            n, edges = _build_lift(m, voltages)
            if _verify_girth_edges(n, edges, g):
                return n, edges
    return None


def construct(k, g):
    """Construct a k-regular graph with girth >= g.
    Returns (n, edges)."""
    deadline = _DEADLINE
    if k == 3 and g == 16:
        # Primary: Thue-Morse voltage lift of Petersen (small n if it works)
        try:
            result = _find_thue_morse_lift(k, g, deadline)
            if result is not None:
                n, edges = result
                if _verify_girth_edges(n, edges, g):
                    return n, edges
        except Exception:
            pass
        # Guaranteed-valid fallback: LUW construction
        try:
            return _luw_construct(k, g)
        except Exception:
            pass
    # General case
    return _luw_construct(k, g)
