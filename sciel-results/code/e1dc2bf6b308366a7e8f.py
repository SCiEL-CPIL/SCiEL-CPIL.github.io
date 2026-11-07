"""
Search for maximum edge-disjoint simple cycles on a fixed vertex set of the 21-vertex host.

Uses a backtracking search with strong pruning (connectivity, degree, edge-disjointness)
to find up to 10 edge-disjoint cycles all using the same vertex set.
"""

# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {
    'n': 21,
    'edges': [
        [0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20],
        [1, 2], [1, 3], [1, 6], [1, 17], [1, 20],
        [2, 3], [2, 4], [2, 7], [2, 18],
        [3, 4], [3, 5], [3, 8], [3, 19],
        [4, 5], [4, 6], [4, 9], [4, 20],
        [5, 6], [5, 7], [5, 10],
        [6, 7], [6, 8], [6, 11],
        [7, 8], [7, 9], [7, 12],
        [8, 9], [8, 10], [8, 13],
        [9, 10], [9, 11], [9, 14],
        [10, 11], [10, 12], [10, 15],
        [11, 12], [11, 13], [11, 16],
        [12, 13], [12, 14], [12, 17],
        [13, 14], [13, 15], [13, 18],
        [14, 15], [14, 16], [14, 19],
        [15, 16], [15, 17], [15, 20],
        [16, 17], [16, 18],
        [17, 18], [17, 19],
        [18, 19], [18, 20],
        [19, 20]
    ]
}


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_cycle(vset, adj, used_edges, start, timeout, vlist_sorted):
    """
    Find a simple cycle on the vertex set vset using only edges not in used_edges.
    Returns a list of vertices forming the cycle, or None.
    """
    k = len(vset)
    if k < 3:
        return None

    vs = set(vset)
    deadline = time.time() + timeout

    # Quick degree feasibility
    for v in vs:
        deg = 0
        for nb in adj[v]:
            if nb in vs and (min(v, nb), max(v, nb)) not in used_edges:
                deg += 1
        if deg < 2:
            return None

    path = [start]
    visited = {start}
    sys.setrecursionlimit(10000)

    def _connected(unvisited, last):
        if not unvisited:
            return True
        to_visit = set()
        for nb in adj[last]:
            if nb in unvisited and (min(last, nb), max(last, nb)) not in used_edges:
                to_visit.add(nb)
        if not to_visit:
            return False
        seen = set(to_visit)
        stack = list(to_visit)
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if nb in unvisited and nb not in seen and (min(cur, nb), max(cur, nb)) not in used_edges:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == len(unvisited)

    def dfs(cur):
        if time.time() > deadline:
            return None
        if len(path) == k:
            if start in adj[cur] and (min(cur, start), max(cur, start)) not in used_edges:
                return list(path)
            return None

        unvisited = vs - visited
        if not _connected(unvisited, cur):
            return None

        cands = []
        for nxt in adj[cur]:
            if nxt not in visited and nxt in vs:
                e = (min(cur, nxt), max(cur, nxt))
                if e in used_edges:
                    continue
                if len(path) == k - 1:
                    if start not in adj[nxt] or (min(nxt, start), max(nxt, start)) in used_edges:
                        continue
                    cands.append((0, 0, nxt))
                else:
                    onward = 0
                    rem_deg = 0
                    for nn in adj[nxt]:
                        if nn in unvisited and nn != cur:
                            e2 = (min(nxt, nn), max(nxt, nn))
                            if e2 not in used_edges:
                                onward += 1
                    for nn in adj[nxt]:
                        if nn in vs and nn != cur:
                            e2 = (min(nxt, nn), max(nxt, nn))
                            if e2 not in used_edges:
                                rem_deg += 1
                    cands.append((onward, rem_deg, nxt))
        cands.sort()
        for _, _, nxt in cands:
            e = (min(cur, nxt), max(cur, nxt))
            if e in used_edges:
                continue
            visited.add(nxt)
            path.append(nxt)
            res = dfs(nxt)
            if res is not None:
                return res
            path.pop()
            visited.remove(nxt)
        return None

    return dfs(start)


def _max_cycles_for(vset, adj, used_edges, max_c, budget):
    """
    Greedily extract up to max_c edge-disjoint cycles on the vertex set vset.
    Returns list of cycles found.
    """
    cycles = []
    t0 = time.time()
    vs = set(vset)
    vlist = list(vset)

    for _ in range(max_c):
        if time.time() - t0 > budget:
            break

        # Try starts in order of remaining degree
        def rem_deg(v):
            return sum(1 for nb in adj[v] if nb in vs and (min(v, nb), max(v, nb)) not in used_edges)

        starts = sorted(vlist, key=rem_deg)
        found = None
        per_try = min(3.0, max(0.5, budget - (time.time() - t0)))
        for st in starts:
            if time.time() - t0 > budget:
                break
            found = _find_cycle(vset, adj, used_edges, st, per_try, vlist)
            if found is not None:
                break

        if found is None:
            break

        # Verify and mark edges
        ok = True
        for i in range(len(found)):
            u, v = found[i], found[(i + 1) % len(found)]
            e = (min(u, v), max(u, v))
            if e in used_edges:
                ok = False
                break
            used_edges.add(e)
        if not ok:
            break
        cycles.append(found)

    return cycles


def _induced_min_deg(vset, adj):
    m = 10**9
    for v in vset:
        d = sum(1 for nb in adj[v] if nb in vset)
        if d < m:
            m = d
    return m


def _induced_edge_count(vset, edges):
    c = 0
    for u, v in edges:
        if u in vset and v in vset:
            c += 1
    return c


def _finalize(cycles, n):
    """Validate cycles and convert to tuples."""
    out = []
    seen_edges = set()
    vset = None
    for cyc in cycles[:10]:
        cyc = list(cyc)
        if len(cyc) < 3:
            continue
        if len(set(cyc)) != len(cyc):
            continue
        if vset is None:
            vset = set(cyc)
        elif set(cyc) != vset:
            continue
        ok = True
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (min(u, v), max(u, v))
            if e in seen_edges:
                ok = False
                break
            seen_edges.add(e)
        if not ok:
            continue
        out.append(tuple(cyc))
    if not out:
        out = [tuple(range(n))]
    return out


def construct():
    t0 = time.time()
    DEADLINE = 85

    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    all_edges_set = set((min(u, v), max(u, v)) for u, v in edges)

    best_cycles = []
    best_count = 0

    full_vset = list(range(n))
    full_set = set(full_vset)

    # Compute theoretical upper bound for full set
    md = _induced_min_deg(full_set, adj)
    ec = _induced_edge_count(full_set, edges)
    upper_full = min(md // 2, ec // n)

    # Primary: full vertex set
    used = set()
    budget = min(20.0, max(1.0, DEADLINE - (time.time() - t0)))
    cycles = _max_cycles_for(full_vset, adj, used, upper_full + 1, budget)
    if len(cycles) > best_count:
        best_count = len(cycles)
        best_cycles = cycles

    if best_count >= upper_full:
        return _finalize(best_cycles, n)

    # Try subsets if time remains
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 3:
        deg = {v: len(adj[v]) for v in range(n)}
        verts_by_deg = sorted(range(n), key=lambda v: -deg[v])

        tried = {frozenset(full_vset)}

        # High-degree core subsets
        for size in range(12, 5, -1):
            if time.time() - t0 > DEADLINE - 1:
                break
            core = verts_by_deg[:size]
            fs = frozenset(core)
            if fs in tried:
                continue
            tried.add(fs)
            sset = set(core)
            md_s = _induced_min_deg(sset, adj)
            if md_s < 2:
                continue
            ec_s = _induced_edge_count(sset, edges)
            if ec_s < 2 * size:
                continue
            ub = min(md_s // 2, ec_s // size)
            if ub <= best_count:
                continue
            b = min(3.0, max(0.5, DEADLINE - (time.time() - t0)))
            used_s = set()
            cyc_s = _max_cycles_for(core, adj, used_s, ub + 1, b)
            if len(cyc_s) > best_count:
                best_count = len(cyc_s)
                best_cycles = cyc_s

        # Consecutive windows
        for start in range(n):
            if time.time() - t0 > DEADLINE - 1:
                break
            for size in range(6, min(13, n - start + 1)):
                window = list(range(start, start + size))
                fs = frozenset(window)
                if fs in tried:
                    continue
                tried.add(fs)
                sset = set(window)
                md_s = _induced_min_deg(sset, adj)
                if md_s < 2:
                    continue
                ec_s = _induced_edge_count(sset, edges)
                if ec_s < 2 * size:
                    continue
                ub = min(md_s // 2, ec_s // size)
                if ub <= best_count:
                    continue
                b = min(2.0, max(0.3, DEADLINE - (time.time() - t0)))
                used_s = set()
                cyc_s = _max_cycles_for(window, adj, used_s, ub + 1, b)
                if len(cyc_s) > best_count:
                    best_count = len(cyc_s)

    return _finalize(best_cycles, n)


# EVOLVE-BLOCK-END
