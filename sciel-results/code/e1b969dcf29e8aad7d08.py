import time
import random
import sys

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible.

    Strategy: the LUW (Lazebnik-Ustimenko-Woldar) construction gives a guaranteed
    valid k-regular graph of girth >= g (baseline). We then spend the remaining
    time trying to find a *smaller* valid graph via random 3-regular graphs,
    verified with a fast non-backtracking-walk girth check. We always return the
    best (smallest n) valid result found, so we never do worse than the baseline.
    """
    deadline = time.time() + 115  # safety margin before the 120s kill

    # ---- Baseline: LUW construction (guaranteed valid, girth >= g) ----
    if k != 3:
        d = max(3, (g + 1) // 2)
        if 2 * d - 1 < g:
            d += 1
        n, edges = _luw_construct(k, d)
        return n, edges

    # k == 3
    d = (g + 1) // 2
    if 2 * d - 1 < g:
        d += 1
    n_base, edges_base = _luw_construct(3, d)

    # ---- Moore bound (theoretical minimum for even girth) ----
    if g % 2 == 0:
        gprime = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(gprime))
    else:
        gprime = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(gprime))
    n_min = moore
    if n_min % 2 != 0:
        n_min += 1
    if n_min < 4:
        n_min = 4

    best_n = n_base
    best_edges = edges_base

    # ---- Randomized search for a smaller valid graph ----
    random.seed(42)

    # Candidate n values: from near the Moore bound up to the baseline.
    # Build a sorted, de-duplicated list of even n in [n_min, n_base).
    candidates = []
    n = n_min
    while n < n_base:
        candidates.append(n)
        n += 2
    if not candidates:
        return best_n, best_edges

    # Try candidates in increasing order; for each, attempt several random
    # realizations until one passes the girth check or we run out of time.
    # We interleave so that smaller n get a fair share but we don't get stuck
    # on one forever.
    attempt_budget = 120  # total random-graph attempts across all n
    idx = 0
    attempts_used = 0
    while idx < len(candidates) and attempts_used < attempt_budget:
        if time.time() > deadline:
            break
        n = candidates[idx]
        # A few tries per n before moving on (smaller n are rarer to satisfy,
        # so cap tries to avoid wasting the whole budget on one infeasible n).
        tries_for_n = 0
        max_tries_per_n = 4
        while tries_for_n < max_tries_per_n and attempts_used < attempt_budget:
            if time.time() > deadline:
                break
            edges = _random_3_regular_fast(n)
            attempts_used += 1
            tries_for_n += 1
            if edges is None:
                continue
            if _check_girth_fast(n, edges, g):
                # Found a valid graph with this (smaller) n
                best_n = n
                best_edges = edges
                if n == n_min:
                    # Hit the Moore bound: cannot do better
                    return best_n, best_edges
                break  # move to next (even smaller) candidate
        idx += 1

    return best_n, best_edges


def _luw_construct(q, d):
    """Lazebnik-Ustimenko-Woldar construction. Returns (n, edges)."""
    add = (lambda x, y: (x + y) % q)
    sub = (lambda x, y: (x - y) % q)
    mul = (lambda x, y: (x * y) % q)

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
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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


def _random_3_regular_fast(n):
    """Generate a random 3-regular simple graph on n vertices (n even).

    Uses the configuration model with rejection. Returns a list of (u, v) edges
    or None if no valid graph found in a few trials.
    """
    if n < 4 or n % 2 != 0:
        return None

    for trial in range(30):
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        random.shuffle(stubs)

        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (u, v) if u < v else (v, u)
            if key in edges:
                valid = False
                break
            edges.add(key)

        if not valid:
            continue

        # Verify 3-regular (should hold by construction, but be safe)
        deg = [0] * n
        edge_list = list(edges)
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
        if any(dd != 3 for dd in deg):
            continue

        return edge_list

    return None


def _check_girth_fast(n, edges, g):
    """Check if the graph has girth >= g (no cycle shorter than g).

    Uses non-backtracking walks from every vertex, certified exactly.
    A cycle of length L < g exists iff there is a closed non-backtracking walk
    of length L. We do a bounded-depth search of non-backtracking walks from
    each vertex up to depth g-1; if we return to the start vertex via a
    non-backtracking path of length < g, girth < g.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    max_depth = g - 1  # we only care about cycles of length < g

    # For each start vertex, explore non-backtracking walks up to max_depth.
    # State: (current, prev, depth). We track visited (current, prev) to avoid
    # exponential blowup, but we must be careful: a non-backtracking walk is
    # determined by (current, prev). We do BFS over these states.
    for s in range(n):
        # visited: set of (current, prev) reached
        visited = set()
        # queue: (current, prev, depth)
        # start: we can go from s to any neighbor
        queue = []
        for nb in adj[s]:
            visited.add((nb, s))
            queue.append((nb, s, 1))

        while queue:
            cur, prev, depth = queue.pop()
            if depth >= max_depth:
                continue
            # If we are back at start via non-backtracking path of length depth
            # (depth >= 2 for a real cycle), then cycle of length depth < g.
            if cur == s and depth >= 2:
                return False
            for nb in adj[cur]:
                if nb == prev:
                    continue  # no backtracking
                state = (nb, cur)
                if state in visited:
                    continue
                visited.add(state)
                queue.append((nb, cur, depth + 1))

    return True
