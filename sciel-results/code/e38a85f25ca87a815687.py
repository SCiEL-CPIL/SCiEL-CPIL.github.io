"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

# EVOLVE-BLOCK-START
_DEADLINE = 120.0

def _verify_girth_fast(n, edges, k, g, budget=30.0):
    """Return True iff the graph is k-regular and has girth >= g.
    Uses non-backtracking BFS from every vertex with optimized state tracking."""
    t0 = time.time()
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Check regularity and simplicity
    for u in range(n):
        if len(adj[u]) != k:
            return False
        if len(set(adj[u])) != k:
            return False
    # Non-backtracking BFS: state is (current, previous)
    # We track the minimum depth at which each state is reached
    for start in range(n):
        if time.time() - t0 > budget:
            return False
        # BFS over non-backtracking states
        # Use a dict mapping (cur, prev) -> depth
        visited = {}
        q = deque()
        for nb in adj[start]:
            state = (nb, start)
            visited[state] = 1
            q.append(state)
        while q:
            cur, prev = q.popleft()
            depth = visited[(cur, prev)]
            if depth >= g:
                break
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == start and depth + 1 < g:
                    return False
                state = (nb, cur)
                if state not in visited:
                    visited[state] = depth + 1
                    q.append(state)
    return True

def construct(k, g):
    t_start = time.time()
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() - t_start > _DEADLINE - 40:
            # Deadline guard: return what we have so far (may be incomplete/invalid)
            return len(order), edges
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    
    # Verify the graph with more time budget
    if time.time() - t_start < _DEADLINE - 20:
        if _verify_girth_fast(n, edges, k, g, budget=_DEADLINE - 25):
            return n, edges
    
    # If verification fails or times out, check regularity and return
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    regular = all(len(adj[u]) == k for u in range(n))
    
    if regular:
        return n, edges
    
    # Last resort: return empty graph (will score 0 but won't crash)
    return 0, []
# EVOLVE-BLOCK-END
