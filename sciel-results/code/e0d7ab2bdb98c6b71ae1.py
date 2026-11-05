"""Rewritten: random 3-regular graph + girth certificate + repair, deadline-guarded."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
_DEADLINE_FRACTION = 0.90  # stop searching this far before the 150s wall-clock kill

def _moore_bound(k, g):
    # Lower bound on n for a k-regular graph of girth g (Moore bound).
    if g % 2 == 0:
        # even girth: n >= 2 * (k-1)^(g/2 - 1) / (1 - (k-1)) ... standard:
        # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
        s = 0
        p = 1
        for i in range(g // 2):
            s += p
            p *= (k - 1)
        return 2 * s
    else:
        # odd girth: n >= (k-1)*(sum_{i=0}^{(g-1)/2 - 1} (k-1)^i) + sum_{i=0}^{(g-1)/2} (k-1)^i
        s1 = 0
        p = 1
        for i in range((g - 1) // 2):
            s1 += p
            p *= (k - 1)
        s2 = 0
        p = 1
        for i in range((g - 1) // 2 + 1):
            s2 += p
            p *= (k - 1)
        return (k - 1) * s1 + s2


def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_valid(k, g, n, edges):
    # simple + k-regular + no cycle shorter than g (non-backtracking BFS from every vertex)
    if n <= 0:
        return False
    adj = [set() for _ in range(n)]
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or v < 0 or u >= n or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        if len(adj[u]) != k:
            return False
    # non-backtracking BFS: state (v, prev). Shortest non-backtracking closed walk
    # from a vertex equals the girth. We stop when we can't go deeper than (g-1) steps.
    max_steps = g - 1
    for start in range(n):
        # queue entries: (node, prev, depth). We track best depth reached per (node,prev).
        visited = {}
        queue = []
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited[(nb, start)] = 1
        head = 0
        while head < len(queue):
            node, prev, depth = queue[head]
            head += 1
            if depth >= max_steps:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                key = (nb, node)
                if key in visited:
                    # we found a non-backtracking walk from start back to (nb,node)
                    # that closes a cycle of length depth+1? Actually a return to a
                    # previously-seen state means a cycle of length <= depth+1.
                    if depth + 1 < g:
                        return False
                    continue
                if depth + 1 < g:
                    visited[key] = depth + 1
                    queue.append((nb, node, depth + 1))
    return True


def _find_short_cycle(k, g, n, adj):
    # Return a list of vertices forming a cycle of length < g, or None.
    # Non-backtracking BFS from a vertex; when we revisit a state we get a cycle.
    for start in range(n):
        # dist[(node,prev)] = depth
        dist = {}
        parent = {}
        queue = []
        for nb in adj[start]:
            queue.append((nb, start))
            dist[(nb, start)] = 1
            parent[(nb, start)] = (start, -1)
        head = 0
        while head < len(queue):
            node, prev = queue[head]
            head += 1
            depth = dist[(node, prev)]
            if depth >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                key = (nb, node)
                if key in dist:
                    # cycle length = depth + 1
                    clen = depth + 1
                    if clen < g:
                        # reconstruct cycle
                        cyc = []
                        cur = (node, prev)
                        while cur != (nb, node):
                            cyc.append(cur[0])
                            cur = parent[cur]
                        cyc.append(nb)
                        # cyc is a path; the cycle is cyc reversed plus start? Build simply:
                        # We have a non-backtracking walk start -> ... -> node -> nb where
                        # (nb,node) was reached earlier with a shorter path. The cycle is the
                        # segment between the two occurrences. Simpler: just return the set.
                        return list(set(cyc))
                    continue
                dist[key] = depth + 1
                parent[key] = (node, prev)
                queue.append((nb, node))
    return None


def _random_k_regular(n, k, rng):
    # Fao-Knuth: random k-regular simple graph. Returns edge list or None on failure.
    if n * k % 2 != 0:
        return None
    # build half-edge list
    while True:
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        rng.shuffle(stubs)
        edges = []
        adj = [set() for _ in range(n)]
        ok = True
        for a in range(0, len(stubs), 2):
            u, v = stubs[a], stubs[a + 1]
            if u == v:
                ok = False
                break
            if v in adj[u]:
                ok = False
                break
            edges.append((u, v))
            adj[u].add(v)
            adj[v].add(u)
        if ok:
            # verify regular
            if all(len(adj[i]) == k for i in range(n)):
                return edges
    # unreachable


def _try_repair(n, k, g, edges, adj, short_cycle, rng):
    # Remove one edge from the short cycle, then reconnect the two exposed stubs to a
    # far-apart vertex pair to preserve 3-regularity and simplicity.
    if not short_cycle:
        return None
    # pick an edge within the cycle set
    cyc_set = set(short_cycle)
    candidates = []
    for u, v in edges:
        if u in cyc_set and v in cyc_set:
            candidates.append((u, v))
    if not candidates:
        return None
    u, v = rng.choice(candidates)
    # remove edge
    new_edges = [e for e in edges if not ((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u))]
    adj[u].discard(v)
    adj[v].discard(u)
    # find a far-apart pair (x,y) with x not adjacent to u, y not adjacent to v,
    # x!=y, and not creating a loop or multi-edge.
    best = None
    for x in range(n):
        if x == u or x == v or x in adj[u]:
            continue
        for y in range(n):
            if y == u or y == v or y == x or y in adj[v]:
                continue
            # ensure x-y not already an edge
            if y in adj[x]:
                continue
            # crude distance heuristic: BFS distance between x and y large
            best = (x, y)
            break
        if best:
            break
    if not best:
        # revert
        return None
    x, y = best
    new_edges.append((u, x))
    new_edges.append((v, y))
    adj[u].add(x)
    adj[v].add(y)
    adj[x].add(u)
    adj[y].add(v)
    return new_edges


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 150.0 * _DEADLINE_FRACTION
    rng = random.Random(12345)

    # Moore bound as a target lower bound
    mb = _moore_bound(k, g)
    # We'll search a range of n values from a reasonable lower bound upward.
    # For k=3,g=16, MB=512. Random 3-regular graphs of girth 16 need n well above MB.
    # Start near a few times MB and grow.

    n_start = max(mb, 2 * k)
    # candidate n values (must satisfy n*k even)
    best = None
    best_n = None

    # Try a sequence of n, from small to large, with limited restarts each.
    n = n_start
    # ensure parity
    if (n * k) % 2 != 0:
        n += 1
    max_n = n_start * 6 + 200
    attempts_per_n = 0

    while n <= max_n and time.time() < deadline:
        # number of restarts for this n: scale with remaining time
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        restarts = max(1, int(remaining * 2))
        for _ in range(restarts):
            if time.time() >= deadline:
                break
            edges = _random_k_regular(n, k, rng)
            if edges is None:
                continue
            adj = _build_adjacency(edges, n)
            # quick validity check (this also certifies girth)
            if _is_valid(k, g, n, edges):
                best = edges
                best_n = n
                # found one; try to improve by going to smaller n? We go upward, so
                # record and break to try next (larger) n only if we want smaller.
                # Since we want MINIMUM n, once we find valid at n, we should try smaller.
                # We'll break and then attempt smaller n in a second phase.
                break
            # attempt a repair
            sc = _find_short_cycle(k, g, n, adj)
            if sc is not None:
                repaired = _try_repair(n, k, g, edges, adj, sc, rng)
                if repaired is not None:
                    adj2 = _build_adjacency(repaired, n)
                    if _is_valid(k, g, n, repaired):
                        best = repaired
                        best_n = n
                        break
        # move to next n
        if best is not None:
            # we found a valid graph at best_n; try to see if a smaller n works.
            break
        n += 2 if (k % 2 == 0) else 2  # keep n*k even: step by 2 is safe
        if (n * k) % 2 != 0:
            n += 1

    # Second phase: try to reduce n below best_n if we found one.
    if best is not None and best_n > mb:
        target = best_n
        lower = max(mb, 2 * k)
        # binary-ish search downward
        lo, hi = lower, target - 2
        while lo <= hi and time.time() < deadline:
            mid = (lo + hi) // 2
            if (mid * k) % 2 != 0:
                mid += 1
            if mid < lo:
                lo = mid
            found = None
            remaining = deadline - time.time()
            if remaining < 2.0:
                break
            restarts = max(1, int(remaining * 1.5))
            for _ in range(restarts):
                if time.time() >= deadline:
                    break
                edges = _random_k_regular(mid, k, rng)
                if edges is None:
                    continue
                adj = _build_adjacency(edges, mid)
                if _is_valid(k, g, mid, edges):
                    found = edges
                    break
                sc = _find_short_cycle(k, g, mid, adj)
                if sc is not None:
                    repaired = _try_repair(mid, k, g, edges, adj, sc, rng)
                    if repaired is not None and _is_valid(k, g, mid, repaired):
                        found = repaired
                        break
            if found is not None:
                best = found
                best_n = mid
                hi = mid - 2
            else:
                lo = mid + 2

    # Fallback: if we never found a valid graph, return a valid but large one.
    # Use a guaranteed-valid construction: the original algebraic family is too big to
    # recompute quickly; instead, do a longer random search at a larger n.
    if best is None:
        # last-ditch: try larger n with more restarts until deadline
        n = max_n + 2
        while time.time() < deadline:
            if (n * k) % 2 != 0:
                n += 1
            edges = _random_k_regular(n, k, rng)
            if edges is not None:
                adj = _build_adjacency(edges, n)
                if _is_valid(k, g, n, edges):
                    best = edges
                    best_n = n
                    break
            n += 4

    if best is None:
        # Absolute fallback: return a trivial valid k-regular graph if possible.
        # For k=3, n must be even and >=4. A 3-regular graph on 4 vertices is K4 (girth 3).
        # That won't satisfy girth g=16, but we must return something. Return a
        # disconnected union of K4's? Not k-regular girth-g. We return best effort.
        # Build a cycle of length n with chords to make 3-regular (girth small).
        n = max(4, mb)
        if (n * k) % 2 != 0:
            n += 1
        edges = _random_k_regular(n, k, rng)
        if edges is None:
            # K4 blocks
            edges = []
            for i in range(0, n, 4):
                a, b, c, d = i, i + 1, i + 2, i + 3
                edges += [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        return n, edges

    return best_n, best
# EVOLVE-BLOCK-END
