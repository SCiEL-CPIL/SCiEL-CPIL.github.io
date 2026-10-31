"""
Construct a k-regular graph with girth at least g.
Primary: voltage lift of Petersen graph over Z_m with deterministic quadratic-residue voltages.
Secondary: systematic circulant search C_n(1,a,b) with algebraic pre-filters.
Fallback: LUW construction.
"""
import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Deterministic voltage lift of Petersen over Z_m with quadratic-residue voltages
        result = _petersen_voltage_lift_deterministic(3, 16, deadline)
        if result is not None:
            return result

        # Strategy 2: Systematic circulant C_n(1,a,b) search
        if not check_deadline():
            result2 = _find_circulant_systematic(3, 16, deadline)
            if result2 is not None:
                return result2

        # Strategy 3: LUW fallback
        for d in [9, 11, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)


def _petersen_voltage_lift_deterministic(k, g, deadline):
    """
    Deterministic voltage lift of Petersen graph over Z_m.
    Edge voltages assigned as distinct quadratic residues mod m to structurally
    prevent short closed walks. Scan m upward from Moore bound.
    """
    # Petersen graph: 10 vertices, 15 edges, 3-regular
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    # Moore bound: n >= 510, so m >= 51 (n = 10*m)
    # n must be such that 10*m is the vertex count
    min_m = 51
    max_m = 95  # n up to 950

    best_result = None
    best_n = float('inf')

    for m in range(min_m, max_m + 1):
        if time.time() > deadline - 5:
            break

        # Try multiple deterministic voltage assignment families
        found = False
        for family in range(6):
            if time.time() > deadline - 3:
                break

            voltages = _assign_voltages_quadratic(m, family)

            # Check girth via non-backtracking BFS on base graph with voltage tracking
            if _lift_girth_check(petersen_adj, voltages, m, g):
                n = 10 * m
                if n < best_n:
                    edges = _build_lift_edges(petersen_edges, voltages, m)
                    if len(edges) == 15 * m:
                        best_n = n
                        best_result = (n, edges)
                    found = True
                    break  # Found valid for this m, move to next m

        if found:
            # We found a valid graph at this m; since we scan upward, this is minimal
            return best_result

    return best_result


def _assign_voltages_quadratic(m, family):
    """
    Assign deterministic voltages to Petersen edges using quadratic residues.
    Different families use different mappings to break symmetries.
    """
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    # Compute quadratic residues mod m
    # For prime m, QRs are {x^2 mod m : x in 1..(m-1)/2}
    # For composite m, use all x^2 mod m for x in 1..m-1
    if m > 2:
        qrs = sorted(set((x * x) % m for x in range(1, m)))
    else:
        qrs = [0]

    # Ensure we have at least 15 distinct values; pad with non-residues if needed
    if len(qrs) < 15:
        # Add small offsets
        extra = []
        val = 1
        while len(qrs) + len(extra) < 15 and val < m:
            if val not in qrs:
                extra.append(val)
            val += 1
        qrs = qrs + extra

    # Select 15 distinct values for the 15 edges
    # Use a deterministic selection that spreads values
    selected = []
    step = max(1, len(qrs) // 15)
    for i in range(15):
        idx = (i * step) % len(qrs)
        selected.append(qrs[idx])

    # Ensure all 15 are distinct
    seen = set()
    unique_selected = []
    for v in selected:
        if v not in seen:
            unique_selected.append(v)
            seen.add(v)
    # Pad if needed
    for v in qrs:
        if len(unique_selected) >= 15:
            break
        if v not in seen:
            unique_selected.append(v)
            seen.add(v)
    while len(unique_selected) < 15:
        unique_selected.append(unique_selected[-1] + 1)

    selected = unique_selected[:15]

    # Apply family-specific permutation to break symmetry
    if family == 0:
        perm = list(range(15))
    elif family == 1:
        perm = [(i * 7) % 15 for i in range(15)]
    elif family == 2:
        perm = [(i * 11) % 15 for i in range(15)]
    elif family == 3:
        perm = [(i * 13 + 5) % 15 for i in range(15)]
    elif family == 4:
        perm = [(i * 17 + 3) % 15 for i in range(15)]
    else:
        perm = [(i * 19 + 7) % 15 for i in range(15)]

    # Deduplicate perm (ensure it's a valid permutation by using a simple shuffle)
    # Actually, let's just use different starting offsets for the QR selection
    voltages = {}
    for idx, (u, v) in enumerate(petersen_edges):
        vol = selected[idx] % m
        voltages[(u, v)] = vol
        voltages[(v, u)] = (-vol) % m

    return voltages


def _assign_voltages_quadratic(m, family):
    """
    Improved deterministic voltage assignment using quadratic residues.
    Each family uses a distinct structural rule.
    """
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    # Compute all quadratic residues mod m
    qrs = set()
    for x in range(1, m):
        qrs.add((x * x) % m)
    qrs_sorted = sorted(qrs)

    # Non-residues
    all_vals = set(range(m))
    non_qrs = sorted(all_vals - qrs)

    voltages = {}

    if family == 0:
        # Outer cycle edges get QRs, inner cycle gets QRs offset, spokes get non-QRs
        outer = petersen_edges[:5]
        inner = petersen_edges[5:10]
        spokes = petersen_edges[10:15]

        for i, (u, v) in enumerate(outer):
            vol = qrs_sorted[i % len(qrs_sorted)] if qrs_sorted else (i + 1) % m
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m

        for i, (u, v) in enumerate(inner):
            vol = qrs_sorted[(i + 3) % len(qrs_sorted)] if qrs_sorted else (i + 2) % m
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m

        for i, (u, v) in enumerate(spokes):
            if non_qrs:
                vol = non_qrs[i % len(non_qrs)]
            else:
                vol = (i * 2 + 1) % m
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m

    elif family == 1:
        # Spread QRs across all edges with different offsets
        for i, (u, v) in enumerate(petersen_edges):
            if qrs_sorted:
                vol = qrs_sorted[(i * 3) % len(qrs_sorted)]
            else:
                vol = (i + 1) % m
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m

    elif family == 2:
        # Use QRs for outer+spokes, non-QRs for inner
        outer = petersen_edges[:5]
        inner = petersen_edges[5:10]
        spokes = petersen_edges[10:15]

        for i, (u, v) in enumerate(outer):
            vol = (qrs_sorted[i * 2 % len(qrs_sorted)] if qrs_sorted else i + 1) % m
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m

        for i, (u, v) in enumerate(inner):
            if non_qrs:
                vol = non_qrs[i % len(non_qrs)]
            else:
                vol = (i * 3 + 1) % m
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m

        for i, (u, v) in enumerate(spokes):
            if qrs_sorted:
                vol = qrs_sorted[(i * 2 + 5) % len(qrs_sorted)]
            else:
                vol = (i + 6) % m
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m

    elif family == 3:
        # Interleaved QR and non-QR
        for i, (u, v) in enumerate(petersen_edges):
            if i % 2 == 0 and qrs_sorted:
                vol = qrs_sorted[(i // 2) % len(qrs_sorted)]
            elif non_qrs:
                vol = non_qrs[(i // 2) % len(non_qrs)]
            else:
                vol = (i + 1) % m
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m

    elif family == 4:
        # QRs with multiplicative structure
        for i, (u, v) in enumerate(petersen_edges):
            base = qrs_sorted[i % len(qrs_sorted)] if qrs_sorted else (i + 1)
            vol = (base * (i + 1)) % m
            if vol == 0:
                vol = 1
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m

    else:
        # Family 5: QRs with additive perturbation
        for i, (u, v) in enumerate(petersen_edges):
            base = qrs_sorted[i % len(qrs_sorted)] if qrs_sorted else (i + 1)
            vol = (base + i * 7) % m
            if vol == 0:
                vol = 1
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m

    return voltages


def _lift_girth_check(petersen_adj, voltages, m, g):
    """
    Check if voltage lift has girth >= g.
    A cycle of length L in the lift = closed non-backtracking walk of length L
    in base with total voltage 0 mod m.
    """
    for start in range(10):
        if _lift_bfs_from(start, petersen_adj, voltages, m, g):
            return False
    return True


def _lift_bfs_from(start, petersen_adj, voltages, m, g):
    """
    Non-backtracking BFS from 'start' tracking voltage.
    Returns True if short cycle found.
    """
    max_len = g - 1
    visited = set()
    queue = deque()

    start_state = (start, -1, 0)
    visited.add(start_state)
    queue.append((start, -1, 0, 0))

    while queue:
        u, prev, vol, depth = queue.popleft()

        if depth >= max_len:
            continue

        for v in petersen_adj[u]:
            if v == prev:
                continue

            new_vol = (vol + voltages[(u, v)]) % m
            new_state = (v, u, new_vol)

            if new_state in visited:
                continue

            visited.add(new_state)
            new_depth = depth + 1

            if v == start and new_vol == 0 and new_depth >= 2:
                if new_depth < g:
                    return True

            if new_depth < max_len:
                queue.append((v, u, new_vol, new_depth))

    return False


def _build_lift_edges(petersen_edges, voltages, m):
    """Build the edge list for the voltage lift."""
    edges = []
    for u, v in petersen_edges:
        vol = voltages[(u, v)]
        for i in range(m):
            a = u * m + i
            b = v * m + ((i + vol) % m)
            if a < b:
                edges.append((a, b))
            else:
                edges.append((b, a))
    return edges


def _find_circulant_systematic(k, g, deadline):
    """
    Systematic search for C_n(1,a,b) with girth >= g.
    Vertex-transitive: check girth from vertex 0 only.
    Algebraic pre-filters for 4-cycles and 6-cycles.
    """
    min_n = 512
    max_n = 940

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 3:
            return None

        half = n // 2
        a_candidates = list(range(max(3, n // 4), half))
        a_candidates.sort(key=lambda a: abs(a - n // 3))

        for a in a_candidates:
            if time.time() > deadline - 3:
                return None

            two_a = (2 * a) % n
            if two_a in (0, 1, n - 1):
                continue

            b_start = a + 1
            b_end = half

            for b in range(b_start, b_end):
                if time.time() > deadline - 3:
                    return None

                if _has_4cycle_fast(n, a, b):
                    continue

                if _has_6cycle_fast(n, a, b):
                    continue

                if _girth_check_circ(n, a, b, g):
                    edges = []
                    for i in range(n):
                        for shift in (1, a, b):
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return None


def _has_4cycle_fast(n, a, b):
    """Check if C_n(1,a,b) has a 4-cycle."""
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]

    for i in range(6):
        g1 = gens[i]
        for j in range(6):
            g2 = gens[j]
            if g2 == (n - g1) % n:
                continue
            s12 = (g1 + g2) % n
            for k in range(6):
                g3 = gens[k]
                if g3 == (n - g2) % n:
                    continue
                s123 = (s12 + g3) % n
                g4 = (n - s123) % n
                if g4 == (n - g3) % n:
                    continue
                if g4 in gens:
                    v1 = g1
                    v2 = s12
                    v3 = s123
                    if v1 != 0 and v2 != 0 and v3 != 0 and v1 != v2 and v2 != v3 and v1 != v3:
                        return True
    return False


def _has_6cycle_fast(n, a, b):
    """Check if C_n(1,a,b) has a 6-cycle using BFS to depth 3."""
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]

    dist = {0: 0}
    parent = {0: -1}
    level = [0]

    for depth in range(3):
        next_level = []
        for u in level:
            for g in gens:
                v = (u + g) % n
                if v == 0 and depth < 2:
                    continue
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level

    # Check for edges between vertices at depth 2 that share a neighbor at depth 3
    # This would form a 6-cycle: 0 -> p(u1) -> u1 -> v -> u2 -> p(u2) -> 0
    depth2_vertices = [v for v in dist if dist[v] == 2]
    depth3_neighbors = {}
    for v in level:  # depth 3
        for g in gens:
            w = (v + g) % n
            if w in dist and dist[w] == 2:
                if w not in depth3_neighbors:
                    depth3_neighbors[w] = []
                depth3_neighbors[w].append(v)

    for u1 in depth2_vertices:
        for u2 in depth2_vertices:
            if u1 >= u2:
                continue
            # Check if u1 and u2 share a common neighbor at depth 3
            common = set(depth3_neighbors.get(u1, [])) & set(depth3_neighbors.get(u2, []))
            if common:
                # Check parents are different (otherwise it's a 4-cycle)
                if parent[u1] != parent[u2]:
                    return True

    return False


def _girth_check_circ(n, a, b, g):
    """
    Non-backtracking BFS from vertex 0 in C_n(1,a,b).
    Returns True if girth >= g.
    """
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]

    best_depth = {}
    queue = deque()

    start = (0, -1)
    best_depth[start] = 0
    queue.append(start)

    max_len = g - 1

    while queue:
        u, prev = queue.popleft()
        d = best_depth[(u, prev)]

        if d >= max_len:
            continue

        for g_step in gens:
            v = (u + g_step) % n
            if v == prev:
                continue

            ns = (v, u)
            nd = d + 1

            if nd > max_len:
                continue

            if ns not in best_depth or nd < best_depth[ns]:
                best_depth[ns] = nd

                if v == 0 and u != -1:
                    if nd < g:
                        return False
                else:
                    queue.append(ns)

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
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
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


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False

    return True


def _bfs_short_cycle(src, adj, n, g):
    """
    BFS from src to find if there's a cycle shorter than g.
    Returns True if a short cycle is found.
    """
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])

    max_depth = g // 2

    while queue:
        u = queue.popleft()
        depth = dist[u]

        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue

        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True

    return False

# EVOLVE-BLOCK-END
