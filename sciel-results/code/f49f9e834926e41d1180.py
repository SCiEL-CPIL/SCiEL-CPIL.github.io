import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _canonical(cycle):
    if len(cycle) < 3:
        return None
    m = len(cycle)
    best = None
    for k in range(m):
        rot = cycle[k:] + cycle[:k]
        rev = list(reversed(rot))
        cand = rot if rot < rev else rev
        if best is None or cand < best:
            best = cand
    return tuple(best)


def _generate_cycle_pool(adj, n, edge_set, rng, time_limit):
    """Generate a pool of simple cycles (canonical tuples of vertices)."""
    start_time = time.time()
    pool = []
    seen = set()

    def _add(cycle):
        if len(cycle) < 3:
            return
        c = _canonical(cycle)
        if c is None:
            return
        # verify edges exist
        m = len(c)
        for i in range(m):
            a = c[i]
            b = c[(i + 1) % m]
            if tuple(sorted((a, b))) not in edge_set:
                return
        if c not in seen:
            seen.add(c)
            pool.append(list(c))

    # Strategy 1: many randomized DFS walks, harvest any back-edge cycle
    def _random_walk_cycle(start):
        # iterative DFS with random neighbor order; record parent to extract cycle
        visited = {start}
        parent = {start: None}
        stack = [start]
        while stack:
            node = stack[-1]
            nbrs = [v for v in adj[node] if v in visited or True]
            # try to find a visited neighbor (back edge) first
            back = None
            for v in adj[node]:
                if v in visited and v != parent.get(node):
                    back = v
                    break
            if back is not None:
                # extract cycle from back .. node
                cyc = [back]
                cur = node
                while cur != back:
                    cyc.append(cur)
                    cur = parent[cur]
                cyc.append(back)
                return cyc
            # else pick an unvisited neighbor
            unvisited = [v for v in adj[node] if v not in visited]
            if not unvisited:
                stack.pop()
                continue
            rng.shuffle(unvisited)
            nxt = unvisited[0]
            visited.add(nxt)
            parent[nxt] = node
            stack.append(nxt)
        return None

    # Strategy 2: DFS from a start, force return to start to get a start-based cycle
    def _start_cycle(start):
        visited = {start}
        parent = {start: None}
        path = [start]
        stack = [(start, iter(list(adj[start])))]
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if nxt == start and len(path) >= 3:
                    return path[:]
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = node
                    path.append(nxt)
                    stack.append((nxt, iter(list(adj[nxt]))))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if path:
                    path.pop()
                # mark back? keep visited to allow other branches (simple path constraint)
        return None

    # Generate a batch
    for _ in range(20000):
        if time.time() - start_time > time_limit:
            break
        # pick a start weighted by degree
        degs = [len(adj[v]) for v in range(n)]
        total = sum(degs)
        if total == 0:
            break
        r = rng.random() * total
        acc = 0
        start = n - 1
        for v in range(n):
            acc += degs[v]
            if r <= acc:
                start = v
                break

        if rng.random() < 0.5:
            cyc = _random_walk_cycle(start)
        else:
            cyc = _start_cycle(start)
        if cyc:
            _add(cyc)
        if len(pool) >= 3000:
            break

    # Strategy 3: enumerate all short cycles (length 3 and 4) deterministically
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # triangles
            for w in adj[v]:
                if w > v and w in adj[u]:
                    _add([u, v, w])
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w <= v:
                    continue
                if w in adj[u]:
                    continue
                for z in adj[w]:
                    if z <= w:
                        continue
                    if z in adj[u]:
                        _add([u, v, w, z])

    return pool


def _solve_packing(pool, edge_set, n, rng, time_limit):
    """Select max number of edge-disjoint cycles from pool. Returns list of selected cycles."""
    start_time = time.time()
    m = len(pool)
    if m == 0:
        return []

    # map edge -> list of cycle indices containing it
    edge_to_cyc = defaultdict(list)
    for i, cyc in enumerate(pool):
        L = len(cyc)
        for k in range(L):
            a = cyc[k]
            b = cyc[(k + 1) % L]
            e = tuple(sorted((a, b)))
            edge_to_cyc[e].append(i)

    # cycles that are fully present in edge_set (they always are by construction)
    selected = []
    used_edge = set()
    used_cyc = set()

    # Greedy: repeatedly pick a cycle using fewest already-used edges (0 preferred)
    # Use a scoring approach with randomization and multiple passes.
    # Precompute cycle edge lists
    cyc_edges = []
    for cyc in pool:
        L = len(cyc)
        es = []
        for k in range(L):
            a = cyc[k]
            b = cyc[(k + 1) % L]
            es.append(tuple(sorted((a, b))))
        cyc_edges.append(es)

    # We'll do a greedy set-packing with a priority that prefers long cycles
    # and few conflicts. Iterate with different random seeds / orderings.
    best_selected = []
    best_len = -1

    for trial in range(40):
        if time.time() - start_time > time_limit:
            break
        tstart = time.time()
        used = set()
        sel = []
        # order cycles: longer first, with random tie-break
        order = list(range(m))
        # weight = (length, random)
        keys = [(-len(pool[i]), rng.random()) for i in range(m)]
        order.sort(key=lambda i: keys[i])
        for i in order:
            if time.time() - tstart > time_limit - (time.time() - start_time):
                break
            if i in used:
                continue
            es = cyc_edges[i]
            if any(e in used for e in es):
                continue
            for e in es:
                used.add(e)
            sel.append(i)

        # local improvement: try to swap a selected short cycle with an unselected long one
        # (simple 2-opt on the packing)
        improved = True
        for _ in range(3):
            if time.time() - start_time > time_limit:
                break
            improved = False
            sel_set = set(sel)
            used = set()
            for i in sel:
                for e in cyc_edges[i]:
                    used.add(e)
            # candidate cycles not selected
            cand = [i for i in range(m) if i not in sel_set]
            rng.shuffle(cand)
            for ci in cand:
                ces = cyc_edges[ci]
                if any(e in used for e in ces):
                    continue
                # find a selected cycle that shares an edge with ci; if swapping gains length
                # Actually ci has no conflict, so adding it is free -> just add
                # But adding may block others; keep simple: add if it doesn't conflict
                # (it doesn't) so add it
                sel.append(ci)
                sel_set.add(ci)
                for e in ces:
                    used.add(e)
                improved = True
                if len(sel) > best_len:
                    pass
            # Now try removing a short selected cycle to free edges for a longer unselected
            # Build map edge->selected cycles
            # Keep it light
            if not improved:
                break

        if len(sel) > best_len:
            best_len = len(sel)
            best_selected = sel[:]

    result = [pool[i] for i in best_selected]
    return result


def _solve_one(edges, n, deadline):
    if not edges:
        return []

    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[b := v].add(u)

    # time budget split
    t0 = time.time()
    pool_time = min(8.0, max(1.0, (deadline - t0) * 0.5))
    pack_time = min(10.0, max(1.0, (deadline - t0) * 0.5))

    rng = random.Random(777)
    pool = _generate_cycle_pool(adj, n, edge_set, rng, pool_time)

    selected = _solve_packing(pool, edge_set, n, rng, pack_time)

    # Build pieces: selected cycles + leftover edges as singletons
    used = set()
    for cyc in selected:
        L = len(cyc)
        for k in range(L):
            a = cyc[k]
            b = cyc[(k + 1) % L]
            used.add(tuple(sorted((a, b))))

    pieces = [list(c) for c in selected]
    for e in edge_set:
        if e not in used:
            pieces.append(list(e))

    return pieces


def construct():
    start_time = time.time()
    total_budget = 145.0
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # give each graph a slice; ensure last finishes before budget
        remaining = total_budget - (time.time() - start_time)
        per = remaining / (len(specs) - i)
        graph_deadline = time.time() + per
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
