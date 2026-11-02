import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    # Small known cages for k=3
    if k == 3:
        if g <= 3:
            return 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        elif g <= 4:
            return 6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
        elif g <= 5:
            edges = []
            for i in range(5):
                edges.append((i, (i+1)%5))
                edges.append((5+i, 5+(i+2)%5))
                edges.append((i, 5+i))
            return 10, edges
        elif g <= 6:
            # Heawood graph: 3-regular, girth 6, n=14
            # It's the point-line incidence graph of Fano plane
            edges = []
            # 7 points, 7 lines, each point on 3 lines, each line has 3 points
            # Fano plane lines: {0,1,2},{0,3,4},{0,5,6},{1,3,5},{1,4,6},{2,3,6},{2,4,5}
            fano_lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
            for i, line in enumerate(fano_lines):
                for p in line:
                    edges.append((p, 7+i))
            return 14, edges
        elif g <= 7:
            # Petersen has girth 5, need girth 7: use (3,7)-cage = 24 vertices
            # The (3,7)-cage is the Foster graph or similar. Actually n(3,7)=24
            # Let me use LUW for this
            pass

    if k == 5 and g == 9:
        # Try the known 1116-vertex cage via voltage cover with search
        n, edges = _try_voltage_cover_cage()
        if n == 1116 and _verify_girth_fast(n, edges, 9):
            return n, edges

        # Fallback to LUW construction
        for d in [4, 5, 6]:
            if time.time() > deadline - 15:
                break
            n, edges = _luw_construct(5, d)
            if _verify_girth_fast(n, edges, 9):
                return n, edges

    # General case: LUW construction
    min_d = max(3, g - 5)
    if (g - 5) % 2 != 0:
        min_d = max(3, g - 4)

    for d in range(min_d, min_d + 8):
        if time.time() > deadline - 10:
            break
        n, edges = _luw_construct(k, d)
        if _verify_girth_fast(n, edges, g):
            return n, edges

    d_fallback = min_d + 8
    return _luw_construct(k, d_fallback)


def _try_voltage_cover_cage():
    """
    Try to construct the 1116-vertex (5,9)-cage using voltage cover of K_6 over Z_186.
    Search for voltage assignments that give girth >= 9.
    """
    n_base = 6
    m = 186
    n_total = n_base * m  # 1116

    # K_6 has 15 edges. We assign a voltage to each directed edge.
    # For edge (i,j) with i<j, voltage v_ij, and v_ji = -v_ij mod m.
    # We need to search over voltage assignments.

    # Use a randomized search with verification
    import random
    rng = random.Random(42)

    # Edge list of K_6
    k6_edges = []
    for i in range(n_base):
        for j in range(i+1, n_base):
            k6_edges.append((i, j))

    # Try several random assignments
    for attempt in range(50):
        voltages = {}
        for (i, j) in k6_edges:
            v = rng.randint(1, m-1)
            voltages[(i, j)] = v
            voltages[(j, i)] = (-v) % m

        # Build the voltage cover
        edges = []
        seen_edges = set()
        for v in range(n_base):
            for w in range(n_base):
                if v == w:
                    continue
                vol = voltages[(v, w)]
                for g_idx in range(m):
                    u_idx = v * m + g_idx
                    g2 = (g_idx + vol) % m
                    w_idx = w * m + g2
                    if u_idx < w_idx:
                        edge_key = (u_idx, w_idx)
                    else:
                        edge_key = (w_idx, u_idx)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(edge_key)

        # Quick girth check
        if _verify_girth_fast(n_total, edges, 9):
            return n_total, edges

    # If random search fails, try a structured assignment
    # Use voltages based on powers of a primitive element
    voltages = {}
    for (i, j) in k6_edges:
        v = ((i+1) * 31 + (j+1) * 7 + 13) % m
        if v == 0:
            v = 1
        voltages[(i, j)] = v
        voltages[(j, i)] = (-v) % m

    edges = []
    seen_edges = set()
    for v in range(n_base):
        for w in range(n_base):
            if v == w:
                continue
            vol = voltages[(v, w)]
            for g_idx in range(m):
                u_idx = v * m + g_idx
                g2 = (g_idx + vol) % m
                w_idx = w * m + g2
                if u_idx < w_idx:
                    edge_key = (u_idx, w_idx)
                else:
                    edge_key = (w_idx, u_idx)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(edge_key)

    if _verify_girth_fast(n_total, edges, 9):
        return n_total, edges

    # Return a dummy that will fail verification
    return 0, []


def _verify_girth_fast(n, edges, g):
    """
    Verify that the graph has girth >= g using BFS from each vertex.
    """
    if n <= 0:
        return False

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Check degree (for the specific task, degree should be k=5)
    # But we don't know k here, so just check it's regular
    if n > 0:
        expected_deg = len(adj[0])
        for i in range(n):
            if len(adj[i]) != expected_deg:
                return False

    # BFS from each vertex to check girth
    # Optimization: only need to go to depth g//2
    max_depth = g // 2

    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])

        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < max_depth:
                        queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True


def _luw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar construction: q-regular bipartite graph
    with girth at least d+5 for odd d.
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
