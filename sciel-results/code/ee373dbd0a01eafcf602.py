"""Construct a 3-regular graph of girth 16 via voltage-lift search over small base graphs."""
import time
import random
from collections import deque

_DEADLINE = 120.0
_DEADLINE_FRAC = 0.88  # keep safety margin

def _moore_bound(k, g):
    if g % 2 == 0:
        r = g // 2
        return k * ((k - 1) ** (r - 1))
    else:
        r = (g - 1) // 2
        return 1 + k * ((k - 1) ** (r - 1))

def _girth_nbt(adj, limit):
    """Return girth (min cycle length) using non-backtracking BFS from every vertex.
    adj: list of lists of neighbour vertex ids.
    limit: only detect cycles up to this length (for speed)."""
    n = len(adj)
    best = limit + 1
    # For each directed edge (u,v), walk forward without backtracking to u
    for u in range(n):
        for v in adj[u]:
            # state: (prev, curr, depth)
            # visited: dict from (prev, curr) -> depth (first time reached)
            # We do BFS on states (prev, curr)
            visited = {}
            queue = deque()
            queue.append((u, v, 1))
            visited[(u, v)] = 1
            while queue:
                prev, curr, d = queue.popleft()
                if d >= limit:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    if nxt == u and d + 1 >= 3:
                        # closed walk
                        if d + 1 < best:
                            best = d + 1
                        continue
                    state = (curr, nxt)
                    nd = d + 1
                    if state in visited:
                        if visited[state] > nd:
                            # shorter path to same state, update
                            visited[state] = nd
                            queue.append((curr, nxt, nd))
                    else:
                        visited[state] = nd
                        queue.append((curr, nxt, nd))
    return best

def _build_lift(base_adj, base_n, m, voltages):
    """Build lifted graph. base_adj[i] = list of (j, direction) where direction 0=out,1=in.
    voltages: dict edge_id -> voltage (for out direction); in uses negative.
    Returns (n, adj) where n = base_n * m."""
    n = base_n * m
    adj = [[] for _ in range(n)]
    for i in range(base_n):
        for (j, dirn) in base_adj[i]:
            # find voltage: store as (to, vol) where vol is for i->j
            pass
    # Redo: use edge list with voltages
    # base_edges: list of (i, j, vol)
    return adj  # placeholder, will be filled by caller

def _search_voltage_lift(base_edges, base_n, m, target_girth, time_limit):
    """Try to find voltages in Z_m such that the lift has girth >= target_girth.
    base_edges: list of (i, j, edge_id) for undirected edges.
    Returns (voltages_list, None) if found, else (None, best_girth_found)."""
    n_edges = len(base_edges)
    start = time.time()
    best_girth = 0
    # Build base adjacency for quick cycle enumeration
    base_adj = [[] for _ in range(base_n)]
    for (i, j, eid) in base_edges:
        base_adj[i].append(j)
        base_adj[j].append(i)
    
    # Enumerate all simple cycles up to length target_girth-1 in base graph
    # Actually, for a lift, a cycle in the lift projects to a closed walk in the base.
    # We need: for every closed walk W of length < target_girth, the sum of voltages != 0 mod m.
    # Closed walks of length L in base correspond to walks (not necessarily simple).
    # This is complex. Alternative: build the lift and check girth directly.
    
    n_lift = base_n * m
    max_iter = 500
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        # Random voltage assignment
        vols = [random.randint(0, m - 1) for _ in range(n_edges)]
        # Build lifted adjacency
        adj = [[] for _ in range(n_lift)]
        for eid, (i, j, _) in enumerate(base_edges):
            v = vols[eid]
            # i -> j with voltage v; j -> i with voltage -v
            for a in range(m):
                u1 = i * m + a
                u2 = j * m + ((a + v) % m)
                adj[u1].append(u2)
            for a in range(m):
                u1 = j * m + a
                u2 = i * m + ((a - v) % m)
                adj[u1].append(u2)
        # Check girth
        g = _girth_nbt(adj, target_girth)
        if g >= target_girth:
            return vols, None
        if g > best_girth:
            best_girth = g
    return None, best_girth

def _cubical_graph_edges():
    """Return edges of the cube graph (Q3): 8 vertices, 12 edges, girth 4, 3-regular."""
    edges = []
    # Vertices 0..7, binary 000..111
    for i in range(8):
        for bit in range(3):
            j = i ^ (1 << bit)
            if i < j:
                edges.append((i, j, 0))
    # Assign edge ids
    for eid in range(len(edges)):
        edges[eid] = (edges[eid][0], edges[eid][1], eid)
    return edges

def _icosahedron_edges():
    """Icosahedron: 12 vertices, 30 edges, 3-regular, girth 5."""
    # Standard icosahedron embedding
    # Using known edge list
    edges = [
        (0,1),(0,2),(0,3),(0,4),(0,5),
        (1,2),(1,6),(1,7),(1,8),(1,5),
        (2,3),(2,9),(2,10),(2,8),(2,5),
        (3,4),(3,11),(3,9),(3,10),
        (4,5),(4,11),(4,6),
        (5,6),(5,7),(5,8),
        (6,7),(6,11),(6,10),
        (7,8),(7,11),(7,9),
        (8,9),(8,10),
        (9,10),(9,11),
        (10,11),
    ]
    # Remove duplicates
    seen = set()
    unique = []
    for (i,j) in edges:
        if i > j:
            i,j = j,i
        if (i,j) not in seen:
            seen.add((i,j))
            unique.append((i,j))
    result = [(i, j, eid) for eid, (i,j) in enumerate(unique)]
    return result

def _petersen_edges():
    """Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5."""
    edges = [
        (0,1),(0,4),(0,5),
        (1,2),(1,6),(1,5),
        (2,3),(2,7),(2,6),
        (3,4),(3,8),(3,7),
        (4,9),(4,8),
        (5,7),(5,9),
        (6,8),(6,9),
        (7,8),
        (8,9),
    ]
    seen = set()
    unique = []
    for (i,j) in edges:
        if i > j:
            i,j = j,i
        if (i,j) not in seen:
            seen.add((i,j))
            unique.append((i,j))
    result = [(i, j, eid) for eid, (i,j) in enumerate(unique)]
    return result

def construct(k, g):
    t0 = time.time()
    deadline = t0 + _DEADLINE * _DEADLINE_FRAC
    
    if k != 3 or g != 16:
        # Fallback: use a simple construction or return empty
        # For generality, return a trivial valid graph if possible
        # This case shouldn't happen per task spec
        return 0, []
    
    mb = _moore_bound(k, g)  # For g=16, k=3: 3 * 2^7 = 384? Wait: g/2=8, so 3*((3-1)^(8-1)) = 3*2^7 = 384
    # Actually Moore bound for even girth g=16: n >= k*((k-1)^((g/2)-1)) = 3*(2^7) = 384
    
    # Try voltage lifts of small 3-regular base graphs
    base_candidates = [
        ("cubical", _cubical_graph_edges(), 8),
        ("icosahedron", _icosahedron_edges(), 12),
        ("petersen", _petersen_edges(), 10),
    ]
    
    # For each base graph, scan m from Moore bound / base_n upward
    best_result = None
    
    for name, edges, base_n in base_candidates:
        min_m = max(2, (mb + base_n - 1) // base_n)
        max_m = min_m + 30  # scan a small range
        for m in range(min_m, max_m + 1):
            if time.time() > deadline:
                break
            # Limit time per (base, m) pair
            alloc = min(5.0, max(0.5, (deadline - time.time()) * 0.1))
            vols, err = _search_voltage_lift(edges, base_n, m, g, alloc)
            if vols is not None:
                # Build the actual graph
                n_lift = base_n * m
                adj = [[] for _ in range(n_lift)]
                for eid, (i, j, _) in enumerate(edges):
                    v = vols[eid]
                    for a in range(m):
                        u1 = i * m + a
                        u2 = j * m + ((a + v) % m)
                        adj[u1].append(u2)
                    for a in range(m):
                        u1 = j * m + a
                        u2 = i * m + ((a - v) % m)
                        adj[u1].append(u2)
                # Build edge list
                edge_set = set()
                edge_list = []
                for u in range(n_lift):
                    for v in adj[u]:
                        if u < v:
                            edge_set.add((u, v))
                edge_list = list(edge_set)
                # Verify degree
                valid = True
                deg = [0] * n_lift
                for (u, v) in edge_list:
                    deg[u] += 1
                    deg[v] += 1
                if any(d != k for d in deg):
                    valid = False
                if valid:
                    if best_result is None or n_lift < best_result[0]:
                        best_result = (n_lift, edge_list)
                break  # found valid for this m, try next base
        if time.time() > deadline:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW-style construction from seed
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q
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
                if side == 0:
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges_list = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_list.append((index[vertex], index[other]))
    return len(order), edges_list
