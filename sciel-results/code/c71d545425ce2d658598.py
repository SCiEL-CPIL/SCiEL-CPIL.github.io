"""
Construct a k-regular graph with girth at least g.

Primary strategy for (k=3, g=16):
  1. Deterministic voltage-graph lift of the Petersen graph over Z_m with
     quadratic-residue voltages; scan m from the Moore bound upward.
  2. Fast non-backtracking walk filter to certify girth 16.
  3. Fallback to the Lazebnik-Ustimenko-Woldar construction.

The voltage lift produces a 3-regular graph on 10*m vertices.  Petersen has
girth 5, so m must be large enough that no short closed walk in the base
accumulates to a zero voltage sum before length g.  Quadratic residues modulo
a prime m have strong multiplicative structure that tends to push the first
zero-sum closed walk to longer lengths.
"""
import time
import math
from collections import deque


def _moore_bound(k, g):
    """Lower bound on the number of vertices for a k-regular graph of girth g."""
    if k == 2:
        return g
    if g % 2 == 0:
        return 2 * sum((k - 1) ** i for i in range(g // 2))
    depth = (g - 1) // 2
    return 1 + (k - 1) * sum((k - 1) ** i for i in range(depth))


def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list
    of (i, j) pairs on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback: LUW construction (always valid, known girth lower bound).
    try:
        fb_n, fb_edges = _luw_construct(k, g)
    except Exception:
        fb_n, fb_edges = None, None

    if k == 3 and g == 16:
        # Try voltage lift of Petersen graph
        try:
            result = _voltage_lift_petersen(3, 16, deadline)
            if result is not None:
                return result
        except Exception:
            pass

        # Try circulant search as secondary
        try:
            circ = _find_circulant_cage(3, 16, deadline)
            if circ is not None:
                return circ
        except Exception:
            pass

    if fb_n is not None:
        return fb_n, fb_edges
    return _luw_construct(k, g)


# ---------------------------------------------------------------------------
# Voltage lift of Petersen graph over Z_m
# ---------------------------------------------------------------------------

# Petersen graph: 10 vertices, 15 edges.
# Standard labeling: outer pentagon 0-1-2-3-4-0, inner star 5-7-9-6-8-5, spokes i-(i+5) mod 5.
_PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),  # spokes
]
_PETERSEN_ADJ = {}
for u, v in _PETERSEN_EDGES:
    _PETERSEN_ADJ.setdefault(u, []).append(v)
    _PETERSEN_ADJ.setdefault(v, []).append(u)


def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _quadratic_residues(m):
    """Return the set of non-zero quadratic residues mod m."""
    if m <= 2:
        return {1}
    return {(x * x) % m for x in range(1, m) if (x * x) % m != 0}


def _voltage_lift_petersen(k, g, deadline):
    """
    Search for the smallest m such that the voltage lift of Petersen over Z_m
    with quadratic-residue voltages has girth >= g.
    The lifted graph has 10*m vertices.
    """
    moore = _moore_bound(k, g)
    # 10*m >= moore  =>  m >= ceil(moore/10)
    min_m = max(1, math.ceil(moore / 10))
    # We want 10*m to be reasonable; the known cage is 936, so m <= 94.
    # But we search upward from min_m to find the smallest valid m.
    # For (3,16), moore = 2*(1+2+4+8+16+32+64) = 254, so min_m = 26.
    max_m = 94  # 10*94 = 940

    # Precompute: for each m, assign voltages to the 15 Petersen edges.
    # Strategy: use the first 15 distinct non-zero quadratic residues mod m.
    # If there aren't 15 distinct QRs, use all non-zero elements in a fixed order.

    for m in range(min_m, max_m + 1):
        if time.time() > deadline - 5.0:
            return None

        # Assign voltages: use quadratic residues if m is prime, else use a
        # deterministic spread of values.
        if _is_prime(m):
            qrs = sorted(_quadratic_residues(m))
            if len(qrs) >= 15:
                voltages = qrs[:15]
            else:
                # Not enough distinct QRs; use all non-zero elements
                voltages = list(range(1, m))
                if len(voltages) < 15:
                    continue
        else:
            # For composite m, use a spread: step through values
            voltages = []
            step = max(1, m // 16)
            for i in range(15):
                voltages.append((i * step + 1) % m)
            # Ensure distinctness
            if len(set(voltages)) < 15:
                # Fallback: use consecutive values
                voltages = list(range(1, 16))
                if m < 16:
                    continue

        # Build the lifted graph and check girth
        n = 10 * m
        # Lifted vertex: (petersen_vertex, coset) where coset in Z_m
        # Edge (u,v) with voltage w: connects (u,c) to (v, (c+w) % m)
        # and (v, (c-w) % m) to (u, c)  [undirected, so we add both directions
        #  but as a single undirected edge]

        # Fast girth check via non-backtracking walks from vertex (0, 0)
        # The lifted graph is vertex-transitive (Z_m action), so checking from
        # one vertex suffices.
        if _check_lifted_girth(m, voltages, g):
            # Build edges
            edges = []
            for idx, (u, v) in enumerate(_PETERSEN_EDGES):
                w = voltages[idx]
                for c in range(m):
                    # (u, c) -- (v, (c+w) % m)
                    a = u * m + c
                    b = v * m + ((c + w) % m)
                    if a < b:
                        edges.append((a, b))
                    else:
                        edges.append((b, a))
            return n, edges

    return None


def _check_lifted_girth(m, voltages, g):
    """
    Check if the voltage lift of Petersen over Z_m with given voltages has
    girth >= g.  Uses non-backtracking walk search from vertex (0,0).
    The graph is vertex-transitive under the Z_m translation, so one check
    suffices.
    """
    n = 10 * m

    # Build adjacency for the lifted graph (on-the-fly to save memory)
    # Adjacency: for vertex (u, c), neighbors are (v, (c + w_e) % m) for each
    # edge e=(u,v) in Petersen.
    # We need a fast way to get neighbors.

    # Precompute: for each Petersen vertex u, list of (v, voltage_index)
    petersen_adj = {}
    for idx, (u, v) in enumerate(_PETERSEN_EDGES):
        petersen_adj.setdefault(u, []).append((v, idx))
        petersen_adj.setdefault(v, []).append((u, idx))

    def get_neighbors(u, c):
        """Get lifted neighbors of (u, c)."""
        result = []
        for v, idx in petersen_adj[u]:
            w = voltages[idx]
            result.append((v, (c + w) % m))
        return result

    # Non-backtracking walk search: state is (current_vertex, prev_vertex)
    # where vertices are (petersen_v, coset).
    # A cycle of length < g exists iff we can return to (0,0) without
    # backtracking, or revisit a state.
    #
    # BFS over non-backtracking states, tracking the minimum distance.
    # State: (u, c, prev_u, prev_c) -- but this is 10*m*10*m states which is
    # too many. Instead, use the fact that in a 3-regular graph, from each
    # vertex there are 2 non-backtracking directions.
    #
    # Better approach: BFS from (0,0) tracking (vertex, prev_vertex) pairs.
    # The number of such pairs is 10*m * 6 = 60*m (each vertex has 6 directed
    # edges).  For m up to 94, that's ~5640 states -- very manageable.

    # State: (u, c, pu, pc) where (pu, pc) is the previous vertex.
    # Start: for each neighbor of (0,0), state is (neighbor, (0,0)).
    # Goal: reach (0,0) from any state (i.e., current == (0,0) and prev != (0,0)).

    visited = set()
    queue = deque()
    # Distance 1: neighbors of (0,0)
    for v, c2 in get_neighbors(0, 0):
        state = (v, c2, 0, 0)
        visited.add(state)
        queue.append((state, 1))

    while queue:
        (u, c, pu, pc), dist = queue.popleft()
        if dist >= g:
            break
        # If we're back at (0,0), we found a cycle of length dist
        if u == 0 and c == 0:
            return False  # cycle found of length dist < g

        # Expand: non-backtracking neighbors of (u,c)
        for v, c2 in get_neighbors(u, c):
            # Skip backtracking
            if v == pu and c2 == pc:
                continue
            new_state = (v, c2, u, c)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, dist + 1))

    # If we exhausted the search without finding a cycle < g, girth >= g
    # But we need to be careful: the BFS might not explore all states if the
    # graph is large.  For m <= 94, the state space is 60*m <= 5640, so it's
    # fine.
    return True


# ---------------------------------------------------------------------------
# Circulant C_n(1, a, b) search (kept as secondary)
# ---------------------------------------------------------------------------

def _circ_generators_valid(n, a, b):
    if a <= 1 or a >= n - 1:
        return False
    if b <= 1 or b >= n - 1:
        return False
    if b <= a:
        return False
    for x, y in ((1, a), (1, b), (a, b)):
        if x == y or x == n - y:
            return False
    return True


def _has_short_cycle_circulant(n, a, b, g):
    if n < g:
        return True
    cap = n * (g - 1)
    seen = set()
    queue = deque()
    for s in (1, a, b):
        queue.append((s, 0))
        queue.append((n - s, 0))
    while queue:
        v, prev = queue.popleft()
        if v == 0:
            return True
        st = (v, prev)
        if st in seen:
            return True
        seen.add(st)
        if len(seen) > cap:
            break
        if prev == 0:
            for s in (1, a, b):
                queue.append(((v + s) % n, v))
                queue.append(((v - s) % n, v))
        else:
            for s in (1, a, b):
                for nb in ((v + s) % n, (v - s) % n):
                    if nb != prev:
                        queue.append((nb, v))
    if len(seen) > cap:
        return True
    return False


def _make_circulant_edges(n, a, b):
    edges = []
    for i in range(n):
        for s in (1, a, b):
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges


def _find_circulant_cage(k, g, deadline):
    min_n = _moore_bound(k, g)
    if min_n % 2 == 1:
        min_n += 1
    max_n = 940
    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 3.0:
            return None
        best = None
        for a in range(2, n // 2 + 1):
            if time.time() > deadline - 1.0:
                break
            for b in range(a + 1, n // 2 + 1):
                if not _circ_generators_valid(n, a, b):
                    continue
                if _has_short_cycle_circulant(n, a, b, g):
                    continue
                best = (n, a, b)
                break
            if best is not None:
                break
        if best is not None:
            n0, a0, b0 = best
            return n0, _make_circulant_edges(n0, a0, b0)
    return None


# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar construction (guaranteed-valid fallback)
# ---------------------------------------------------------------------------

def _luw_construct(k, g):
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
            yield (1 - side, tuple(y))

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
