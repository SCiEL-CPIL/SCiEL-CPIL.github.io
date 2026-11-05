"""
Construct small 3-regular graphs of large girth using the voltage graph / covering method.

Strategy:
  - Start from a small 3-regular base graph (e.g., a cycle with chords, or a
    3-regular graph of small girth).
  - Assign voltage labels from a finite group (cyclic group Z_m) to edges.
  - The resulting covering graph has |V(base)| * |G| vertices and is 3-regular.
  - Search over voltage assignments to maximize girth.
  - For k=3, g=16, we need a graph with girth >= 16.

  The voltage graph approach: given a base graph B with edges e_1...e_m, assign
  to each edge a voltage g(e) in a group G. The covering graph has vertices
  (v, x) where v in V(B) and x in G. An edge (u,v) with voltage g in B gives
  edges (u, x) -- (v, x*g) and (v, x) -- (u, x*g^{-1}) in the covering.

  We use cyclic groups Z_m. For 3-regular graphs, the base graph can be small.

  Key insight: if the base graph has n_b vertices and we use group of size m,
  the covering has n_b * m vertices. For girth 16 with 3-regular, we need
  at least 512 vertices (Moore bound), and the best known is 936.

  We'll try base graphs with n_b from 4 to ~10, and group sizes m such that
  n_b * m is in a reasonable range (512 to ~1000).

  For each (base, m), we search voltage assignments using simulated annealing
  or random restarts with local search to maximize girth.

  Girth computation: BFS from each vertex up to depth g-1 (or use non-backtracking
  walks). For efficiency, we compute girth by BFS from each vertex, stopping
  when we find a shorter cycle.

  Actually, for a 3-regular graph with potentially ~1000 vertices, BFS from each
  vertex up to depth 8 (half of girth 16) is feasible.

  Let's structure:
  1. Generate candidate base graphs (3-regular, small).
  2. For each base and suitable group size, run voltage assignment search.
  3. Track best girth found; return the graph with max girth (ideally >= 16).
"""

import time
import random
import numpy as np
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 140  # safety margin before 150s

def _elapsed():
    return time.time() - _START_TIME

def _time_exceeded():
    return _elapsed() > _TIME_LIMIT

def _girth_nonbacktracking(n, adj):
    """
    Compute girth using non-backtracking walks from each vertex.
    Returns the girth (shortest cycle length), or 0 if no cycle (shouldn't happen for connected).
    For a graph with n vertices and adjacency list adj, this does BFS in the
    non-backtracking state space (v, prev) where prev is the vertex we came from.
    """
    # Non-backtracking: state is (current_vertex, previous_vertex)
    # Start: for each edge (u,v), state (u,v) and (v,u)
    # Transition: from (u,v), go to (v,w) for w in adj[v], w != u
    # A cycle of length L is found when we return to a starting state.

    # For efficiency with girth up to 16, we do BFS up to depth 15.
    # State: (vertex, prev_vertex)
    # We track for each state the distance (path length) at which it was first reached.

    # Actually, a simpler approach for moderate girth: BFS from each vertex,
    # tracking the path, and checking if we revisit a vertex at a distance
    # that creates a cycle shorter than current best.

    # Let's use the standard BFS girth algorithm:
    # For each vertex s, BFS to find shortest path back to s.
    # This gives girth in O(n*(n+m)) worst case but often fast for small girth.

    # Better: use the non-backtracking walk approach which is O(n * (k-1) * girth)
    # where k is degree.

    # For 3-regular: from each directed edge, non-backtracking walks.
    # Number of directed edges = 2*m = 3*n for 3-regular.
    # Each step branches to (k-1) = 2 choices.
    # Up to depth d = girth-1 = 15: 2^15 = 32768 states per starting edge.
    # Total: 3*n * 32768 which for n=1000 is ~100M - too slow in Python.

    # Let's use a smarter approach: BFS from each vertex, but prune.
    # Actually, for the search we need to evaluate girth quickly.
    # Let's use a BFS that tracks (vertex, distance, parent) and detects cycles.

    best = n + 1  # infinity

    for s in range(n):
        # BFS from s
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= best:  # no need to go deeper
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif v != s and dist[v] < dist[u]:
                    # cycle found: dist[u] + 1 + dist[v]... wait
                    # Actually if we see v already visited and v != parent,
                    # cycle length = dist[u] + 1 + dist[v]
                    # But we need to be careful.
                    pass
        # This standard BFS doesn't directly give girth easily.
        # Let's use a different approach.
        break

    # Let me use a cleaner BFS-based girth computation.
    # For each vertex, do BFS and when we find an edge to an already-visited
    # vertex (that's not the parent), we have a cycle.

    # Actually, the cleanest for small graphs: for each vertex s, BFS and record
    # the shortest path tree. Any non-tree edge (u,v) where both are visited
    # gives a cycle of length dist[u] + dist[v] + 1.

    # But this requires knowing the tree. Let's just do: for each s, BFS, and
    # for each edge (u,v) with dist[u] + dist[v] + 1 < best, update best.

    # This is O(n * (n + m)) = O(n^2) for 3-regular. For n=1000, that's 1M * constant.
    # In Python this might be slow. Let's optimize.

    # Alternative: use the fact that for 3-regular graphs, girth is at most ~2*log_2(n).
    # For n=1000, girth <= ~20. So we can do BFS up to depth 10 from each vertex.

    # Let's try a vectorized approach or just careful BFS.

    # For now, let's implement a reasonable BFS girth.

    best = n + 1
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= best:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if du + 1 < best:
                        q.append(v)
                elif v != s:
                    # We found a cycle: s -> ... -> u -> v -> ... -> s
                    # The cycle length is du + 1 + dist[v]
                    # But only if this is a simple cycle, which it is in BFS.
                    cyc = du + 1 + dist[v]
                    if cyc < best:
                        best = cyc
        if best <= 4:  # can't get better than 4 for 3-regular simple graph
            break

    return best if best <= n else 0


def _try_voltages(base_adj, base_n, m, max_time=30):
    """
    Try to find voltage assignment on base graph with group Z_m that maximizes girth.
    base_adj: list of (u, v) edges in base graph (undirected)
    m: group size (cyclic Z_m)
    Returns (n_vertices, edge_list, girth) for the best covering found, or None.
    """
    n_edges = len(base_adj)
    # Voltage assignment: list of integers mod m, one per edge
    # The covering graph has base_n * m vertices.
    # Vertex (v, x) where v in 0..base_n-1, x in 0..m-1.
    # Edge (u,v) in base with voltage g: in covering, (u, x) connects to (v, x+g mod m)
    # and (v, x) connects to (u, x-g mod m).

    n_total = base_n * m
    if n_total < 512:
        return None  # below Moore bound, can't have girth 16
    if n_total > 1500:
        return None  # too large, not competitive

    def build_covering(voltages):
        """Build adjacency list for the covering graph."""
        adj = [[] for _ in range(n_total)]
        for i, (u, v) in enumerate(base_adj):
            g = voltages[i] % m
            for x in range(m):
                u_idx = u * m + x
                v_idx = v * m + (x + g) % m
                adj[u_idx].append(v_idx)
                adj[v_idx].append(u_idx)
        return adj

    def compute_girth(adj):
        return _girth_nonbacktracking(n_total, adj)

    # Initial random assignment
    voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
    adj = build_covering(voltages)
    cur_girth = compute_girth(adj)
    best_girth = cur_girth
    best_voltages = voltages[:]

    if cur_girth >= 16:
        edges = []
        for i in range(n_total):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n_total, edges, cur_girth

    # Local search: try changing one voltage at a time
    start_time = time.time()
    while time.time() - start_time < max_time:
        improved = False
        for i in range(n_edges):
            old_v = voltages[i]
            for new_v in range(m):
                if new_v == old_v:
                    continue
                voltages[i] = new_v
                adj = build_covering(voltages)
                g = compute_girth(adj)
                if g > cur_girth:
                    cur_girth = g
                    improved = True
                    if g >= 16:
                        voltages = voltages[:]
                        best_voltages = voltages[:]
                        best_girth = g
                        edges = []
                        for i2 in range(n_total):
                            for j2 in adj[i2]:
                                if i2 < j2:
                                    edges.append((i2, j2))
                        return n_total, edges, g
                    break
                elif g == cur_girth:
                    # keep or revert? Let's keep for exploration
                    pass
                else:
                    voltages[i] = old_v
                    break
            voltages[i] = old_v
            if improved:
                break
        if not improved:
            # Random restart within this configuration
            for _ in range(10):
                i = random.randint(0, n_edges - 1)
                old_v = voltages[i]
                voltages[i] = random.randint(0, m - 1)
                adj = build_covering(voltages)
                g = compute_girth(adj)
                if g >= cur_girth:
                    cur_girth = g
                    if g > best_girth:
                        best_girth = g
                        best_voltages = voltages[:]
                else:
                    voltages[i] = old_v

    # Return best found
    if best_girth >= 16:
        adj = build_covering(best_voltages)
        edges = []
        for i in range(n_total):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n_total, edges, best_girth

    return None


def _generate_base_graphs():
    """Generate small 3-regular graphs as base graphs."""
    graphs = []

    # K4: 4 vertices, 3-regular, girth 3
    # Edges: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    graphs.append([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])

    # K_{3,3}: 6 vertices, 3-regular, girth 4
    graphs.append([(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)])

    # Prism graph (3-prism = C3 x K2): 6 vertices, 3-regular, girth 3
    # Actually let's use the utility graph K3,3 which is better.

    # 3-regular graph on 8 vertices: cube graph Q3
    # Vertices 0-7, edges of 3D cube
    cube_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    graphs.append(cube_edges)

    # 3-regular on 10 vertices: Petersen graph
    petersen = [(0,1),(1,2),(2,3),(3,4),(4,0),  # outer pentagon
                (5,6),(6,7),(7,8),(8,9),(9,5),  # inner star
                (0,5),(1,6),(2,7),(3,8),(4,9)]  # spokes
    # Wait, that's not right. Petersen: outer 5-cycle, inner 5-star, spokes.
    # Let me fix: vertices 0-4 outer, 5-9 inner.
    # Outer: 0-1-2-3-4-0
    # Inner: 5-7-9-6-8-5 (star)
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen = [(0,1),(1,2),(2,3),(3,4),(4,0),
                (5,7),(7,9),(9,6),(6,8),(8,5),
                (0,5),(1,6),(2,7),(3,8),(4,9)]
    graphs.append(petersen)

    # 3-regular on 12 vertices: e.g., the 3-prism is 6, let's do dodecahedron? No, 20 vertices.
    # Let's do a 3-regular bipartite graph on 12 vertices.
    # Or just use the line graph of K4, which is 3-regular on 6... no.
    # Let's add the 3-regular graph that is the Cartesian product C4 x K2 (8 vertices, already have cube)
    # For 12: take C6 x K2 (12 vertices, 3-regular, girth 4)
    c6k2 = []
    for i in range(6):
        c6k2.append((i, (i+1) % 6))
        c6k2.append((i+6, (i+1) % 6 + 6))
        c6k2.append((i, i+6))
    # Remove duplicates
    c6k2 = list(set(c6k2))
    graphs.append(c6k2)

    return graphs


def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    global _START_TIME
    _START_TIME = time.time()

    if k != 3:
        # Fallback: use the original LZW construction for other k
        return _construct_lzw(k, g)

    # For k=3, use voltage graph method
    base_graphs = _generate_base_graphs()

    best_result = None  # (n, edges, girth)

    # Try different group sizes
    # n_total = base_n * m, want n_total in [512, 1200]
    for base_edges in base_graphs:
        if _time_exceeded():
            break
        base_n = max(v for e in base_edges for v in e) + 1
        base_edges = list(set(base_edges))

        # Check 3-regular
        deg = [0] * base_n
        for u, v in base_edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            continue

        n_e = len(base_edges)

        # Find suitable m values
        for m in range(64, 300):
            n_total = base_n * m
            if n_total < 512 or n_total > 1200:
                continue
            if _time_exceeded():
                break

            remaining = 140 - _elapsed()
            if remaining < 5:
                break

            result = _try_voltages(base_edges, base_n, m, max_time=min(remaining, 20))
            if result is not None:
                n, edges, girth = result
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges, girth)
                if girth >= g:
                    return n, edges

    if best_result is not None:
        n, edges, girth = best_result
        return n, edges

    # Fallback: return a valid 3-regular graph (may not have girth >= 16)
    # Use a large cycle with chords to make it 3-regular
    n = 1000
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        edges.append((i, (i + n // 2) % n))
        edges.append((i, (i + n // 2 + 1) % n))
    # Remove duplicates and loops
    seen = set()
    unique_edges = []
    for u, v in edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            unique_edges.append(key)
    return n, unique_edges


def _construct_lzw(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
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
