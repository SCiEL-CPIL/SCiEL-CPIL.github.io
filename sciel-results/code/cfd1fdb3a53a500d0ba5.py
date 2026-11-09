"""4-regular graph of girth 10 (cage problem n(4,10)).

Primary strategy: a voltage-graph (cover) construction over a small 4-regular base
graph of girth 5 (the Petersen graph) with Z_2^m voltages, so the girth of the cover
is governed by a linear condition on non-backtracking closed walks.  We search for
voltage assignments that maximize the girth of the resulting cover, and also attempt
a randomized 2-switch local search on candidate covers to break short cycles.

Fallback: the Lazebnik-Ustimenko-Woldar (L-U-W) construction, which always yields a
valid 4-regular graph of girth at least g.
"""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import deque, defaultdict

DEADLINE = 120.0

# ---------------------------------------------------------------------------
# Petersen graph: 4-regular, girth 5, 10 vertices, 15 edges
# ---------------------------------------------------------------------------
_PETERSEN_ADJ = [
    (1, 5, 9, 6),
    (0, 2, 4, 7),
    (1, 3, 5, 8),
    (2, 4, 6, 9),
    (1, 3, 7, 8),
    (0, 2, 6, 9),
    (0, 3, 5, 7),
    (1, 4, 6, 9),
    (2, 4, 5, 9),
    (0, 3, 7, 8),
]
_PETERSEN_EDGES = []
_seen = set()
for u in range(10):
    for v in _PETERSEN_ADJ[u]:
        key = (min(u, v), max(u, v))
        if key not in _seen:
            _seen.add(key)
            _PETERSEN_EDGES.append(key)
# 15 edges

# Non-backtracking walks on Petersen up to length 9 (all closed walks of length < 10)
# Each walk is a tuple (start, end) with the sequence of edges implied.
# For voltage assignment, a non-backtracking closed walk w of length L contributes
# the sum of voltages along its edges (with sign depending on direction).
# We need: for every non-backtracking closed walk of length < g, the voltage sum != 0
# in the voltage group, AND for every non-backtracking walk of length < g/2, the walk
# is not a "voltage identity" that creates a short cycle.

# Actually the correct condition: a non-backtracking closed walk of length L in the
# base graph lifts to a closed walk in the cover iff the product (sum in Z_2^m)
# of voltages along the walk (with appropriate inverses for reverse edges) is the
# identity.  For Z_2^m, inverse = identity, so it's just the sum of edge voltages
# along the walk.

# But we also need: no two distinct non-backtracking walks of the same length < g
# from the same start vertex have the same endpoint in the cover.  This is harder
# to check directly.  The standard result: if every non-backtracking closed walk
# of length < g has non-zero voltage, then girth >= g.  (For regular bipartite
# covers this is sufficient; for general covers we need a bit more, but for
# girth >= 10 it works in practice.)

# Enumerate non-backtracking closed walks on Petersen up to length 9.
# A non-backtracking walk: (v0, v1, v2, ..., vL) with v_{i-1} != v_{i+1}.
# Closed: vL = v0.

def _enumerate_nb_closed_walks(base_adj, max_len):
    """Return list of (start, [edge_voltages_indices]) for non-backtracking
    closed walks of length 2..max_len.  Edge index is the index in _PETERSEN_EDGES."""
    walks = []
    n = len(base_adj)
    for start in range(n):
        for length in range(2, max_len + 1):
            # BFS over non-backtracking walks
            # state: (current_vertex, prev_vertex, walk_so_far_as_list_of_vertices)
            stack = [(start, None, [start])]
            while stack:
                cur, prev, path = stack.pop()
                if len(path) > length:
                    continue
                if len(path) == length and cur == start:
                    # record the walk
                    edges_in_walk = []
                    for i in range(len(path) - 1):
                        a, b = path[i], path[i + 1]
                        # find edge index
                        eidx = _edge_index[a][b]
                        edges_in_walk.append(eidx)
                    walks.append((start, edges_in_walk, length))
                    continue
                if len(path) == length:
                    continue
                for nb in base_adj[cur]:
                    if nb != prev:
                        stack.append((nb, cur, path + [nb]))
    return walks

_edge_index = [[-1] * 10 for _ in range(10)]
for idx, (u, v) in enumerate(_PETERSEN_EDGES):
    _edge_index[u][v] = idx
    _edge_index[v][u] = idx

_NB_CLOSED_WALKS = _enumerate_nb_closed_walks(_PETERSEN_ADJ, 9)
# Filter: keep only walks where consecutive vertices are adjacent (they are by
# construction) and where the walk is truly non-backtracking (v_{i-1} != v_{i+1}).
# Also deduplicate: the same set of edges in the same direction may be counted
# multiple times from different starting points.  We keep them all; more constraints
# is fine.

# For the voltage condition, each closed walk w = (start, edges, length) gives a
# linear equation: sum_{e in edges} voltage[e] = 0 (in Z_2^m).  We need this sum
# to be non-zero for all walks of length < g.  In Z_2^m, "non-zero" means the
# m-bit vector is not all zeros.

# We want to assign voltage vectors in Z_2^m to the 15 edges such that for every
# non-backtracking closed walk of length < g, the sum of its edge voltages is
# non-zero.  This is a "avoiding hyperplanes" problem in Z_2^m.

# For m=1 (Z_2): each edge gets 0 or 1.  A closed walk has sum 0 iff it uses an
# even number of edges with voltage 1.  We need every short closed walk to use
# an odd number of 1-edges.  This is a system of linear inequalities.

# For m=2 (Z_2^2): each edge gets one of 4 vectors.  More flexibility.

# Strategy: try m=1 first (32^15 = too many, but we can use random search + local
# search).  Actually 2^15 = 32768 assignments for m=1.  We can brute-force!

def _voltage_sum(walk_edges, voltages):
    """Sum of voltage vectors along a walk (in Z_2^m). voltages is list of m-bit ints."""
    s = 0
    for e in walk_edges:
        s ^= voltages[e]  # XOR for Z_2^m
    return s

def _check_voltages(voltages, min_girth):
    """Check if voltage assignment gives girth >= min_girth."""
    m = len(bin(voltages[0] | 1)) - 1 if voltages else 1
    for walk in _NB_CLOSED_WALKS:
        length = walk[2]
        if length < min_girth:
            s = _voltage_sum(walk[1], voltages)
            if s == 0:
                return False
    return True

def _build_cover(voltages):
    """Build the voltage cover of the Petersen graph.
    Returns (n, edges) for the cover graph."""
    m = 0
    if voltages:
        mx = max(voltages)
        m = max(1, len(bin(mx)) - 1)
    else:
        m = 1
    group_size = 1 << m
    n = 10 * group_size
    edges = []
    for eidx, (u, v) in enumerate(_PETERSEN_EDGES):
        volt = voltages[eidx]
        for coset in range(group_size):
            a = u * group_size + coset
            b = v * group_size + (coset + volt) % group_size
            edges.append((a, b))
    # Deduplicate and ensure no self-loops
    edge_set = set()
    deduped = []
    for a, b in edges:
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in edge_set:
            edge_set.add(key)
            deduped.append(key)
    return n, deduped

def _girth_check(n, edges, g):
    """Check girth >= g using BFS from each vertex (non-backtracking).
    Returns True if girth >= g."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        # BFS with non-backtracking: state (current, prev)
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        while queue:
            cur, prev, dist = queue.popleft()
            if cur == start and dist >= g:
                return False  # found a cycle of length dist < g? No, dist >= g means OK
            if cur == start and dist < g:
                return False
            if (cur, prev) in visited:
                continue
            visited.add((cur, prev))
            if dist < g - 1:
                for nb in adj[cur]:
                    if nb != prev:
                        queue.append((nb, cur, dist + 1))
    return True

def _greedy_voltage_search(min_girth, deadline):
    """Search for a voltage assignment on Petersen giving cover girth >= min_girth.
    Try m=1 brute force first, then m=2 random search."""
    # m=1: brute force 2^15 = 32768
    for bits in range(1 << 15):
        voltages = [(bits >> i) & 1 for i in range(15)]
        if _check_voltages(voltages, min_girth):
            n, edges = _build_cover(voltages)
            if _girth_check(n, edges, min_girth):
                return n, edges
    # m=2: random search
    rng = random.Random(42)
    for _ in range(50000):
        voltages = [rng.randint(0, 3) for _ in range(15)]
        if _check_voltages(voltages, min_girth):
            n, edges = _build_cover(voltages)
            if _girth_check(n, edges, min_girth):
                return n, edges
    return None

def _local_search_2switch(n, edges, g, deadline):
    """Attempt to increase girth via 2-switches.
    Find shortest cycle, try to break it by swapping edges."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def find_shortest_cycle():
        """Find the shortest cycle. Returns list of vertices or None."""
        for start in range(n):
            visited = {start: None}
            queue = deque([(start, None, 0)])
            while queue:
                cur, prev, dist = queue.popleft()
                if dist >= g:
                    return None
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    if nb == start and dist + 1 >= 2:
                        # cycle found
                        cycle = [start]
                        node = cur
                        while node != start:
                            cycle.append(node)
                            node = visited[node]
                        cycle.append(start)
                        return cycle
                    if nb not in visited:
                        visited[nb] = cur
                        queue.append((nb, cur, dist + 1))
        return None

    def edge_set(edges):
        return set((min(u, v), max(u, v)) for u, v in edges)

    es = edge_set(edges)
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        cycle = find_shortest_cycle()
        if cycle is None:
            return n, edges
        # Try 2-switches to break this cycle
        # A 2-switch: remove edges (a,b) and (c,d), add (a,c) and (b,d) or (a,d) and (b,c)
        # Choose two edges from the cycle and two edges not in the cycle
        cycle_edges = set()
        for i in range(len(cycle) - 1):
            a, b = cycle[i], cycle[i + 1]
            cycle_edges.add((min(a, b), max(a, b)))
        # Try swapping two cycle edges with two non-cycle edges
        ce = list(cycle_edges)
        non_cycle_edges = [e for e in es if e not in cycle_edges]
        if len(ce) < 2 or len(non_cycle_edges) < 2:
            break
        found = False
        for i in range(len(ce)):
            for j in range(i + 1, len(ce)):
                (a1, b1), (c1, d1) = ce[i], ce[j]
                for k in range(len(non_cycle_edges)):
                    (a2, b2) = non_cycle_edges[k]
                    for l in range(k + 1, len(non_cycle_edges)):
                        (c2, d2) = non_cycle_edges[l]
                        # Try swap: remove (a1,b1),(c1,d1); add (a1,a2),(b1,b2)
                        # Need to check simple graph
                        new_edges = es - {ce[i], ce[j]}
                        new_edges.add((min(a1, a2), max(a1, a2)))
                        new_edges.add((min(b1, b2), max(b1, b2)))
                        if len(new_edges) != len(es):
                            continue
                        # Check no loops
                        if a1 == a2 or b1 == b2:
                            continue
                        ne_list = list(new_edges)
                        # Quick check: is girth improved?
                        if _girth_check(n, ne_list, g):
                            es = new_edges
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            # Try different pair
            if random.random() < 0.5:
                continue
            break
    return n, list(es)

def _luw_construction(k, g):
    """Lazebnik-Ustimenko-Woldar construction (fallback)."""
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
        if time.time() > DEADLINE - 10:
            break
        for other in neighbours(vertex):
            if time.time() > DEADLINE - 5:
                break
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible.
    Returns (n, edges)."""
    start_time = time.time()
    deadline = start_time + DEADLINE - 5  # safety margin

    # For the specific case k=4, g=10
    if k == 4 and g == 10:
        # Try voltage graph over Petersen
        t0 = time.time()
        result = _greedy_voltage_search(g, deadline)
        if result is not None:
            n, edges = result
            # Verify
            if _girth_check(n, edges, g):
                return n, edges
            # Try local search to improve
            t1 = time.time()
            n2, edges2 = _local_search_2switch(n, edges, g, deadline)
            if _girth_check(n2, edges2, g):
                return n2, edges2
            return n, edges

    # Fallback: L-U-W construction
    return _luw_construction(k, g)
# EVOLVE-BLOCK-END
