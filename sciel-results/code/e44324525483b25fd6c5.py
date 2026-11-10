"""Minimum edge path cover via Gallai's theorem: odd pairing + Eulerian circuit + cut."""
import time
from collections import deque, defaultdict

_DEADLINE = 148.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _bfs_path(adj, s, t, forbidden):
    """Shortest simple path from s to t, internal vertices not in forbidden."""
    if s == t:
        return [s]
    if s in forbidden or t in forbidden:
        return None
    prev = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    if t not in prev:
        return None
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if len(path) >= 3 and set(path[1:-1]) & forbidden:
        return None
    return path


def _pair_odd(adj, odd):
    """Find vertex-disjoint paths pairing up all odd vertices. Returns list of paths."""
    remaining = list(odd)
    used_internal = set()
    paths = []
    # Greedy: repeatedly find the shortest available pair
    while len(remaining) >= 2 and _remaining() > 3.0:
        best = None
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                s, t = remaining[i], remaining[j]
                p = _bfs_path(adj, s, t, used_internal)
                if p is None:
                    continue
                if best is None or len(p) < len(best):
                    best = p
        if best is None:
            break
        used_internal |= set(best[1:-1])
        paths.append(best)
        remaining.remove(best[0])
        remaining.remove(best[-1])
    # Fallback for any leftovers (shouldn't happen in connected graph)
    while len(remaining) >= 2:
        s, t = remaining[0], remaining[1]
        p = _bfs_path(adj, s, t, used_internal)
        if p is None:
            p = _bfs_path(adj, s, t, set())
        if p is None:
            remaining.remove(s)
            remaining.remove(t)
            continue
        used_internal |= set(p[1:-1])
        paths.append(p)
        remaining.remove(s)
        remaining.remove(t)
    return paths


def _eulerian_circuit(adj):
    """Hierholzer's algorithm. Returns (circuit, edge_counts)."""
    if not adj:
        return [], {}
    stack = [next(iter(adj))]
    circuit = []
    while stack:
        v = stack[-1]
        if adj[v]:
            u = next(iter(adj[v]))
            adj[v].discard(u)
            adj[u].discard(v)
            stack.append(u)
        else:
            circuit.append(stack.pop())
    circuit.reverse()
    return circuit, adj


def _solve_one(edges):
    adj = defaultdict(set)
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)

    if not edges:
        return []

    degree = {u: len(adj[u]) for u in adj}
    odd = sorted([u for u in adj if degree[u] % 2 == 1])

    if len(odd) == 0:
        # Eulerian: one circuit
        hadj = defaultdict(set)
        for (u, v) in edges:
            hadj[u].add(v)
            hadj[v].add(u)
        circuit, _ = _eulerian_circuit(hadj)
        return [circuit]

    # Pair odd vertices with vertex-disjoint paths
    pair_paths = _pair_odd(adj, odd)
    k = len(pair_paths)

    # Build multigraph H = G + pair_paths (as multiset of edges)
    # Use a list of edges to handle multiplicities
    edge_list = list(edges)
    for pp in pair_paths:
        for i in range(len(pp) - 1):
            edge_list.append((pp[i], pp[i + 1]))

    # Build adjacency for Hierholzer with edge indices
    hadj = defaultdict(list)
    for idx, (u, v) in enumerate(edge_list):
        hadj[u].append((v, idx))
        hadj[v].append((u, idx))

    # Hierholzer on multigraph
    used_edge = [False] * len(edge_list)
    stack = [(edge_list[0][0], None)]
    circuit = []
    while stack:
        v, in_idx = stack[-1]
        advanced = False
        for u, idx in hadj[v]:
            if not used_edge[idx]:
                used_edge[idx] = True
                hadj[v].remove((u, idx))
                stack.append((u, idx))
                advanced = True
                break
        if not advanced:
            circuit.append(stack.pop())
    circuit.reverse()

    # circuit is a list of vertices; between consecutive vertices, an edge was used.
    # We need to identify which edges are "added" (from pair_paths) to cut.
    # Build a set of added edges with multiplicity.
    from collections import Counter
    added_counter = Counter()
    for pp in pair_paths:
        for i in range(len(pp) - 1):
            added_counter[(pp[i], pp[i + 1])] += 1
            added_counter[(pp[i + 1], pp[i])] += 1

    # Walk the circuit, tracking which added edges we cross.
    # The circuit uses edges in order. We need to map each consecutive vertex pair
    # to an edge index. Rebuild: track edges used in order.
    # Re-run Hierholzer but record the edge sequence.
    used_edge = [False] * len(edge_list)
    stack = [(edge_list[0][0], None)]
    edge_seq = []  # list of (u, v, edge_index) in circuit order
    while stack:
        v, _ = stack[-1]
        advanced = False
        for u, idx in hadj[v]:
            if not used_edge[idx]:
                used_edge[idx] = True
                hadj[v].remove((u, idx))
                edge_seq.append((v, u, idx))
                stack.append((u, idx))
                advanced = True
                break
        if not advanced:
            stack.pop()
    # edge_seq is in reverse-circuit order; reverse it
    edge_seq.reverse()

    # Now edge_seq[i] = (u, v, idx) means the circuit goes u->v using edge_list[idx].
    # Identify added edges (from pair_paths) by checking if (u,v) is in added_counter.
    # Cut at each added edge: the segment between consecutive added edges is a path.
    added_positions = []
    for pos, (u, v, idx) in enumerate(edge_seq):
        if added_counter[(u, v)] > 0:
            added_counter[(u, v)] -= 1
            added_positions.append(pos)

    # The circuit is cyclic. Cut at added_positions.
    # Between added edge at position a and next added edge at position b,
    # the original edges (a+1 .. b-1) form a path segment.
    # Vertices: circuit vertices are edge_seq[i][0] for i, and the last is edge_seq[-1][1].
    # Vertex at position i (before edge i) = edge_seq[i][0]. Vertex after edge i = edge_seq[i][1].
    # A path from after added edge at pos a to before added edge at pos b:
    #   vertices: edge_seq[a][1], edge_seq[a+1][1], ..., edge_seq[b-1][1]
    #   (if a+1 == b, it's just [edge_seq[a][1]] which is a single vertex - skip)

    m = len(edge_seq)
    if not added_positions:
        # Shouldn't happen if k>0
        return [circuit]

    results = []
    for i in range(len(added_positions)):
        a = added_positions[i]
        b = added_positions[(i + 1) % len(added_positions)]
        if b < a:
            b += m
        # Path vertices from after edge a to before edge b
        path_verts = []
        for pos in range(a + 1, b):
            path_verts.append(edge_seq[pos][0])
        path_verts.append(edge_seq[(b - 1) % m][1])
        # Deduplicate consecutive (shouldn't be needed) and ensure simple
        if len(path_verts) >= 2:
            # Verify it's a valid simple path using original edges
            results.append(path_verts)

    # Verify and fix: ensure all edges covered exactly once, paths simple.
    # Build coverage check
    from collections import Counter
    cover = Counter()
    for p in results:
        for i in range(len(p) - 1):
            cover[(p[i], p[i + 1])] += 1
            cover[(p[i + 1], p[i])] += 1

    # Check all original edges covered
    orig_counter = Counter()
    for (u, v) in edges:
        orig_counter[(u, v)] += 1
        orig_counter[(v, u)] += 1

    # If coverage is off, fall back to greedy longest path
    if cover != orig_counter:
        remaining_edges = set(edges)
        final_paths = []
        while remaining_edges and _remaining() > 2.0:
            radj = defaultdict(set)
            for (u, v) in remaining_edges:
                radj[u].add(v)
                radj[v].add(u)
            verts = sorted(radj.keys(), key=lambda x: len(radj[x]))
            if not verts:
                break
            start = verts[0]
            path = [start]
            visited_v = {start}
            cur = start
            while True:
                nxt = None
                for v in sorted(radj[cur]):
                    if v not in visited_v and (min(cur, v), max(cur, v)) in remaining_edges:
                        nxt = v
                        break
                if nxt is None:
                    break
                remaining_edges.discard((min(cur, nxt), max(cur, nxt)))
                path.append(nxt)
                visited_v.add(nxt)
                cur = nxt
            if len(path) >= 2:
                final_paths.append(path)
            else:
                break
        return final_paths

    # Verify simplicity
    clean = []
    for p in results:
        if len(set(p)) == len(p):
            clean.append(p)
    if len(clean) != len(results):
        # Rebuild from scratch via greedy
        remaining_edges = set(edges)
        final_paths = []
        while remaining_edges and _remaining() > 2.0:
            radj = defaultdict(set)
            for (u, v) in remaining_edges:
                radj[u].add(v)
                radj[v].add(u)
            verts = sorted(radj.keys(), key=lambda x: len(radj[x]))
            if not verts:
                break
            start = verts[0]
            path = [start]
            visited_v = {start}
            cur = start
            while True:
                nxt = None
                for v in sorted(radj[cur]):
                    if v not in visited_v and (min(cur, v), max(cur, v)) in remaining_edges:
                        nxt = v
                        break
                if nxt is None:
                    break
                remaining_edges.discard((min(cur, nxt), max(cur, nxt)))
                path.append(nxt)
                visited_v.add(nxt)
                cur = nxt
            if len(path) >= 2:
                final_paths.append(path)
            else:
                break
        return final_paths

    return results


def construct():
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = _instance(*spec)
        paths = _solve_one(edges)
        results.append(paths)
    return results
