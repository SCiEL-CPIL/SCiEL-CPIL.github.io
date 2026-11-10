"""Search for a small 3-regular graph of girth >= g, falling back to LUW construction."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k; returns (n, edges)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
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


def _girth_at_least_nb(adj, n, g, t0=None, deadline=140.0):
    """Return True iff graph (adjacency list) has girth >= g.
    Uses non-backtracking walks from every vertex, stopping at length g-1.
    If any non-backtracking walk of length < g returns to its start, girth < g.
    Early-exits on first violation."""
    if g <= 2:
        return True
    # Non-backtracking walks: state is (current_vertex, prev_vertex)
    # We track all walks of length up to g-1.
    # A walk v0-v1-...-vL is non-backtracking if vi != v(i-2) for all valid i.
    # Girth < g iff there exists a non-backtracking closed walk of length < g.
    # For simplicity and speed, we do BFS over (vertex, prev) states up to depth g-1.
    # A cycle of length L < g exists iff we can reach (v0, v1) from (v1, v0) 
    # in L-1 steps... actually let's use a different approach.
    
    # Better: for each starting vertex s, do a BFS over states (v, prev) 
    # where prev is the vertex we came from (or -1 for start).
    # Track depth. If we reach state (s, u) at depth L where L+1 >= g, 
    # and the path forms a valid cycle, we have a cycle of length L+1.
    # Actually, a non-backtracking closed walk of length L corresponds to 
    # a cycle of length <= L. If L < g, girth < g.
    
    # Most efficient: BFS from each (s, neighbor) state, tracking depth.
    # If we return to s via a non-backtracking path of length < g, fail.
    
    max_depth = g - 1  # we're looking for cycles of length 2..g-1
    
    for s in range(n):
        # BFS over (vertex, prev) states
        # State: (v, p) means we're at v, came from p
        # Start: for each neighbor u of s, state (u, s) at depth 1
        # If we reach (s, u) at depth d where d >= 2 and d < g, 
        #   we have a cycle of length d (s -> ... -> u -> s)
        #   Wait: state (u, s) at depth 1 means s->u.
        #   State (s, u) at depth d means a path of length d from s back to s 
        #   via u as the penultimate vertex. The cycle is s->...->u->s of length d.
        #   We need d < g for a violation.
        
        visited = {}  # (v, p) -> min_depth
        queue = []
        for u in adj[s]:
            state = (u, s)
            visited[state] = 1
            queue.append((u, s, 1))
        
        head = 0
        while head < len(queue):
            v, p, depth = queue[head]
            head += 1
            
            if t0 is not None and (time.time() - t0) > deadline:
                return False  # timeout, be safe
            
            if depth >= max_depth:
                continue  # no need to extend further
            
            nd = depth + 1
            for u in adj[v]:
                if u == p:
                    continue  # non-backtracking
                # Check if we've hit the start vertex s
                if u == s:
                    # Cycle of length nd
                    if nd < g:
                        return False
                    continue
                state = (u, v)
                if state not in visited:
                    visited[state] = nd
                    queue.append((u, v, nd))
    return True


def _random_regular_graph(n, k, rng):
    """Generate a random simple k-regular graph on n vertices (n*k even).
    Returns adjacency list or None on failure."""
    if n * k % 2 != 0 or n < k + 1:
        return None
    # configuration model
    stubs = []
    for v in range(n):
        stubs.extend([v] * k)
    for _attempt in range(200):
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, n * k, 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if ok and all(len(x) == k for x in adj):
            return [sorted(x) for x in adj]
    return None


def _deadline_reached(t0, limit):
    return (time.time() - t0) > limit


def construct(k, g):
    t0 = time.time()
    HARD_LIMIT = 140.0  # safety margin under 150s wall-clock

    # Moore bound lower bound on n
    if g >= 4:
        # For k=3, g even: Moore bound is 2*(2^(g/2-1) - 1) + 2 = 2^r where r = g/2 - 1
        # Actually for 3-regular: n >= 2 + 2*(2^(r-1)-1)*... 
        # Standard: for k-regular, girth g even, n >= 2*(k-1)^(g/2-1) + 2*(k-1)^(g/2-2) ... 
        # Simplified safe lower bound: 2 * (2 ** (g//2 - 1))
        moore = 2 * (2 ** (g // 2 - 1))
    else:
        moore = 0

    rng = random.Random(12345)

    # Build a list of candidate n values: even numbers (n*k must be even -> n even for k odd)
    step = 2 if k % 2 == 1 else 1
    lo = max(moore, k + 1)
    if lo % step:
        lo += step
    
    # Search window: try from Moore bound upward
    # Use the fast non-backtracking checker to enable more attempts
    hi = lo + 600  # wider search window with faster checker

    n = lo
    while n <= hi and not _deadline_reached(t0, HARD_LIMIT):
        attempts_per_n = 0
        # With the faster checker, we can afford more attempts
        max_attempts = 30 if n < 800 else 10
        while attempts_per_n < max_attempts and not _deadline_reached(t0, HARD_LIMIT):
            adj = _random_regular_graph(n, k, rng)
            attempts_per_n += 1
            if adj is None:
                continue
            if _girth_at_least_nb(adj, n, g, t0=t0, deadline=HARD_LIMIT):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        n += step

    # Fallback: guaranteed-valid LUW construction
    try:
        return _luw_construct(k, g)
    except Exception:
        raise
# EVOLVE-BLOCK-END
