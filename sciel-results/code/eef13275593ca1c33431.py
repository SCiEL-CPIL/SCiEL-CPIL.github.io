"""Construct small k-regular graphs of given girth via random greedy search."""
import random
import time

_DEADLINE = 150.0
_T0 = time.time()


def _remaining():
    return _DEADLINE - 2.0 - (time.time() - _T0)


def construct(k, g):
    """Return (n, edges): a k-regular simple graph on n vertices with girth >= g.

    Strategy: random greedy (probabilistic-method) construction. Repeatedly try to
    build a k-regular graph with no cycle shorter than g by greedily adding edges
    while avoiding short cycles. Keep the best (smallest n) valid graph found.
    """
    best = None  # (n, edges)
    best_n = None

    # Moore lower bound for n(k, g): for even girth g=2r, n >= k * (k-1)^(r-1) + 1
    r = g // 2
    moore = k * (k - 1) ** (r - 1) + 1

    # We'll try n values starting near a heuristic upper estimate.
    # Heuristic: random greedy for (k,g) tends to work around the "greedy bound".
    # Start a bit above Moore and grow.
    import math

    def girth_ok(adj, n, g):
        """Return True if no cycle shorter than g. Use BFS non-backtracking from each
        vertex up to depth (g-1)//2 to detect short cycles."""
        maxdepth = (g - 1) // 2
        for s in range(n):
            # BFS storing (node, parent) to avoid backtracking
            from collections import deque
            q = deque()
            q.append((s, -1, 0))
            seen = {s: 0}
            while q:
                u, par, d = q.popleft()
                if d == maxdepth:
                    continue
                for v in adj[u]:
                    if v == par:
                        continue
                    if v in seen:
                        # found a cycle of length d + seen[v] + 1
                        if d + seen[v] + 1 < g:
                            return False
                    else:
                        seen[v] = d + 1
                        q.append((v, u, d + 1))
        return True

    def try_build(n, k, g, rng, max_restarts=200):
        """Random greedy: build k-regular graph avoiding short cycles. Return edges or None."""
        for _ in range(max_restarts):
            if _remaining() < 0.5:
                return None
            # Start with a random near-regular degree sequence via configuration-ish greedy.
            # Use a simple approach: random permutation pairing for k-regular (multigraph-free via check).
            # We'll greedily assign edges to satisfy degree k while avoiding short cycles.
            deg = [0] * n
            adj = [set() for _ in range(n)]
            # Random order of "slots"
            slots = []
            for i in range(n):
                for _ in range(k):
                    slots.append(i)
            rng.shuffle(slots)
            # Greedily match slots into edges
            unmatched = list(range(len(slots)))
            rng.shuffle(unmatched)
            ok = True
            # We'll do a greedy matching of the 2nk slots into nk edges.
            # Represent each slot as (vertex, id). Pair them.
            # Simpler: repeatedly pick a random slot, connect to a random other slot of a
            # different vertex with available degree, avoiding short cycles.
            remaining = {i: k for i in range(n)}
            # order vertices by degree need
            for _ in range(n * k // 2):
                if _remaining() < 0.5:
                    return None
                # pick vertex with max remaining
                a = max(range(n), key=lambda i: remaining[i])
                if remaining[a] == 0:
                    break
                # candidate b
                cands = [i for i in range(n) if i != a and remaining[i] > 0 and i not in adj[a]]
                if not cands:
                    ok = False
                    break
                rng.shuffle(cands)
                placed = False
                for b in cands:
                    # tentatively add edge, check girth locally (only need to check cycles
                    # through this edge quickly). Full check is expensive; do a light check:
                    # ensure no new cycle < g by BFS from a and b bounded.
                    adj[a].add(b)
                    adj[b].add(a)
                    remaining[a] -= 1
                    remaining[b] -= 1
                    # light girth check: BFS limited to depth g//2 from a and b avoiding backtrack
                    if not _local_girth_ok(adj, n, g, a, b):
                        # revert
                        adj[a].remove(b)
                        adj[b].remove(a)
                        remaining[a] += 1
                        remaining[b] += 1
                        continue
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if not ok:
                continue
            if any(remaining[i] != 0 for i in range(n)):
                continue
            # full verification
            if girth_ok(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return edges
        return None

    def _local_girth_ok(adj, n, g, a, b):
        """Check no cycle shorter than g using the new edge a-b. BFS from a up to depth
        (g-2)//2 avoiding b as first step and vice versa; if b reachable from a within
        that depth via a path not using edge a-b, a short cycle forms."""
        from collections import deque
        maxdepth = (g - 2) // 2
        # BFS from a, not using edge a-b, depth <= maxdepth; if reach b -> cycle len <= maxdepth+1 < g?
        # cycle length = pathlen(a->b) + 1. Need pathlen+1 < g => pathlen < g-1 => pathlen <= g-2.
        # We bound pathlen <= maxdepth = (g-2)//2 which is stricter; sufficient (may miss some but safe-ish).
        # To be safe use depth g-2.
        maxdepth = g - 2
        q = deque([(a, -1, 0)])
        while q:
            u, par, d = q.popleft()
            if d >= maxdepth:
                continue
            for v in adj[u]:
                if v == par:
                    continue
                if u == a and v == b:
                    continue  # don't use the new edge forward
                if v == b:
                    # path length d+1 from a to b not using edge a-b -> cycle d+1+1
                    if d + 1 + 1 < g:
                        return False
                    continue
                q.append((v, u, d + 1))
        return True

    # Search n values. Start from a heuristic and grow.
    # Heuristic greedy bound for (k,g): n ~ k*(k-1)^((g-2)/2) is Moore; random greedy needs a bit more.
    import math
    base = int(math.ceil(k * (k - 1) ** ((g - 2) / 2)))
    # try a few candidate sizes
    candidates = []
    for factor in [1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0]:
        candidates.append(int(base * factor) + 1)
    candidates = sorted(set(candidates))

    rng = random.Random(12345)
    for n in candidates:
        if n < moore:
            continue
        if _remaining() < 1.0:
            break
        edges = try_build(n, k, g, rng, max_restarts=60)
        if edges is not None:
            if best_n is None or n < best_n:
                best_n = n
                best = (n, edges)
            # found one at this size; since we go ascending, this is smallest so far.
            break

    # Fallback: if nothing found, use a guaranteed (but large) explicit construction:
    # a circulant-like or LUW-style graph that is k-regular with girth >= g.
    if best is None:
        best = _fallback(k, g)

    return best


def _fallback(k, g):
    """Explicit k-regular graph with girth >= g (large but valid). Use a known
    construction: the (k)-regular graph from a large prime field (LUW) full component.
    """
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
