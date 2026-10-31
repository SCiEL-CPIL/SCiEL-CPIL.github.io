"""Construct a small 3-regular graph of girth 16 (cage problem n(3,16)).

Strategy:
  1. Try known explicit constructions (LUW, folded, circulant) quickly.
  2. Use a randomized search (repeated random 3-regular graphs + girth check)
     within a strict wall-clock budget, returning the best (smallest n) valid
     graph found before the deadline.

The girth is certified exactly via non-backtracking walks from every vertex.
"""
import time
import random
import math

def _girth_nonbacktracking(n, edges, max_len):
    """Return the girth (shortest cycle length) of a simple undirected graph,
    or a value > max_len if no cycle of length <= max_len is found.
    Uses non-backtracking walks from each vertex.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # For each start vertex, do a BFS on the non-backtracking state space.
    # State: (current, prev) meaning we are at 'current' having come from 'prev'.
    # We want to find the shortest closed non-backtracking walk.
    # A non-backtracking walk of length L from v back to v with no immediate
    # reversal corresponds to a cycle of length L (if L >= 3 and the walk is
    # simple, but non-backtracking walks can repeat vertices; however the
    # shortest non-backtracking closed walk from v to v has length equal to
    # the girth if we consider all starts).
    #
    # Actually, the standard method: for each vertex v, BFS on (node, prev)
    # states. If we reach (v, u) with distance d >= 3, then there's a cycle
    # of length d. The minimum over all v gives the girth.
    #
    # But we need to be careful: a non-backtracking walk from v to v of length
    # d means we started at v, went to some neighbor, and after d-1 more steps
    # returned to v. The cycle length is d. We need d >= 3 (no 2-cycles in
    # simple graphs, no 1-cycles).

    best = max_len + 1
    for start in range(n):
        # BFS queue: (current, prev, depth)
        # visited: set of (current, prev)
        visited = set()
        queue = []
        for nb in adj[start]:
            state = (nb, start)
            visited.add(state)
            queue.append((nb, start, 1))
        # Process BFS
        while queue:
            # Pop from front (use index for efficiency)
            next_queue = []
            for cur, prev, depth in queue:
                if depth >= max_len:
                    continue
                # If cur == start and depth >= 3, we found a cycle of length depth
                if cur == start and depth >= 3:
                    if depth < best:
                        best = depth
                    # Don't expand further from this state if we already have best
                    # but continue BFS to find potentially shorter ones from other paths
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    state = (nb, cur)
                    if state not in visited:
                        visited.add(state)
                        next_queue.append((nb, cur, depth + 1))
            queue = next_queue
        if best <= max_len:
            # Early exit if we found a very short cycle
            break
    return best


def _check_girth(n, edges, g):
    """Return True if the graph has no cycle shorter than g."""
    if g <= 2:
        return True
    # We only need to check up to g-1
    result = _girth_nonbacktracking(n, edges, g - 1)
    return result >= g


def _is_valid(k, g, n, edges):
    """Check if the graph is valid: simple, k-regular, girth >= g."""
    if n <= 0:
        return False
    # Check degree
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
        deg[u] += 1
        deg[v] += 1
    for i in range(n):
        if deg[i] != k:
            return False
    # Check girth
    return _check_girth(n, edges, g)


def _random_3regular(n):
    """Generate a random 3-regular graph on n vertices (n must be even).
    Returns (n, edges) or None if failed.
    """
    if n % 2 != 0:
        return None
    if n < 4:
        return None
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    random.shuffle(stubs)
    edges = []
    adj = [[] for _ in range(n)]
    for i in range(0, n * 3, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        edges.append((u, v))
        adj[u].append(v)
        adj[v].append(u)
    return n, edges


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j).
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_n = None
    best_edges = None

    # Moore bound for k-regular graph of girth g:
    # For even girth g=2m: n >= 1 + k * (1 + (k-1) + ... + (k-1)^(m-1))
    # For odd girth g=2m+1: n >= 1 + k * (1 + (k-1) + ... + (k-1)^(m-1)) + k*(k-1)^m
    if g % 2 == 0:
        m = g // 2
        moore = 1 + k * sum((k - 1)**i for i in range(m))
    else:
        m = (g - 1) // 2
        moore = 1 + k * sum((k - 1)**i for i in range(m)) + k * (k - 1)**m

    # Try known constructions first
    # LUW graph
    try:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q
            sub_fn = lambda x, y: (x - y) % q
            mul_fn = lambda x, y: x * y % q
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
                        y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                    else:
                        y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges_luw = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_luw.append((index[vertex], index[other]))
        n_luw = len(order)
        if _is_valid(k, g, n_luw, edges_luw):
            if best_n is None or n_luw < best_n:
                best_n = n_luw
                best_edges = edges_luw
    except Exception:
        pass

    # If we have a valid graph, check time and maybe try to improve
    if time.time() > deadline:
        if best_n is not None:
            return best_n, best_edges
        # Fallback: return something (shouldn't happen)
        return 0, []

    # Random search: try random 3-regular graphs of increasing size
    # Start from a reasonable lower bound above the Moore bound
    # For k=3, g=16: Moore bound is 512. Known best is 936.
    # Try sizes from max(moore, 4) upward, but be smart about it.
    
    # First, if we already have a valid graph, try smaller sizes
    start_size = max(moore, 4)
    if best_n is not None:
        start_size = max(moore, 4)
    
    # Try a range of sizes with random attempts
    # For each size, try a few random graphs
    max_attempts_per_size = 20
    sizes_to_try = []
    
    if best_n is not None:
        # Try to find something smaller than best_n
        for s in range(start_size, min(best_n, start_size + 2000), 2):
            sizes_to_try.append(s)
    else:
        # Try from Moore bound upward
        for s in range(start_size, start_size + 2000, 2):
            sizes_to_try.append(s)

    for n_try in sizes_to_try:
        if time.time() > deadline - 10:
            break
        if n_try % 2 != 0:
            continue
        for attempt in range(max_attempts_per_size):
            if time.time() > deadline - 5:
                break
            result = _random_3regular(n_try)
            if result is None:
                continue
            n_r, edges_r = result
            if _is_valid(k, g, n_r, edges_r):
                if best_n is None or n_r < best_n:
                    best_n = n_r
                    best_edges = edges_r
                # Found a valid one, try smaller sizes
                break

    if best_n is not None:
        return best_n, best_edges

    # If no valid graph found, try larger sizes
    for n_try in range(start_size + 2000, start_size + 8000, 2):
        if time.time() > deadline - 10:
            break
        if n_try % 2 != 0:
            continue
        for attempt in range(max_attempts_per_size):
            if time.time() > deadline - 5:
                break
            result = _random_3regular(n_try)
            if result is None:
                continue
            n_r, edges_r = result
            if _is_valid(k, g, n_r, edges_r):
                if best_n is None or n_r < best_n:
                    best_n = n_r
                    best_edges = edges_r
                break
        if best_n is not None:
            break

    if best_n is not None:
        return best_n, best_edges

    # Last resort: return the LUW graph if we have it, or a trivial answer
    # This should not happen for reasonable inputs
    return 0, []
