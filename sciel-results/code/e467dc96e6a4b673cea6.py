"""Construct small 4-regular graphs of high girth (cage problem n(4,10))."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _is_valid(k, g, n, edges):
    """Check degree, simplicity, and girth via non-backtracking BFS from every vertex."""
    if n <= 0:
        return False
    adj = [[] for _ in range(n)]
    seen = set()
    for a, b in edges:
        if a == b:
            return False
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        key = (min(a, b), max(a, b))
        if key in seen:
            return False
        seen.add(key)
        adj[a].append(b)
        adj[b].append(a)
    for v in range(n):
        if len(adj[v]) != k:
            return False
    # Girth check: BFS in non-backtracking line graph from every (v, -1)
    # A cycle of length L corresponds to a non-backtracking closed walk of length L.
    # We check that no such walk of length < g returns to its start.
    # For efficiency, we do a BFS from each (start, prev) and look for a return to start
    # with the correct non-backtracking condition.
    # Actually, the standard method: for each directed edge (u,v), do BFS in the directed-edge
    # space avoiding backtracking, and check if we can return to (v,u) in < g-1 steps? 
    # Simpler: enumerate all non-backtracking walks of length up to g-1 from each vertex 
    # and check for collisions that form a cycle.
    # Most robust: for each vertex s, BFS in state (current, prev) where prev is the node we came from.
    # We want to detect any closed non-backtracking walk of length L < g.
    # A closed NBW of length L starting and ending at s means: s -> v1 -> ... -> v_{L-1} -> s,
    # with v1 != s (no loops), and vi != v_{i-2} for all i (no backtracking).
    # We can do: for each s, for each neighbor v1 of s, BFS in state (cur, prev) starting at (v1, s),
    # and see if we reach (s, v1) in fewer than g-1 additional steps? 
    # Let's use the cleanest known method:
    # For each starting vertex s, do a BFS in the "non-backtracking" sense.
    # State: (node, prev_node). Start states: (v, s) for each neighbor v of s.
    # We track distance = length of walk so far (starting from s, so (v,s) has length 1).
    # We want to know if any state (s, v) is reached with total length < g.
    # If (s, v) is reached at length L, that means there's a walk s -> ... -> s of length L+1? 
    # Let's be precise.
    # Walk: s = w0, w1, ..., wL = s. NBW means wi != w_{i-2}.
    # Length L. We need L < g.
    # BFS from s: 
    #   Level 1: states (w1, s) for w1 in adj[s]. 
    #   Level 2: from (w1, s), go to (w2, w1) for w2 in adj[w1], w2 != s.
    #   ...
    #   We check if any state (s, w_{L-1}) is reached at level L. 
    #   If so, the walk s->w1->...->w_{L-1}->s has length L and is non-backtracking.
    #   We need to ensure L < g.
    # So we do BFS up to depth g-1 and check for (s, *) at depth L where L < g.
    # Wait, if we reach (s, prev) at depth L, the walk length is L. We need L < g.
    # So we BFS to depth g-1 and check if any state has node == s.
    # But we must exclude the trivial return at depth 2 (s->v->s) which is backtracking.
    # Our BFS already avoids backtracking (w2 != s when coming from (w1,s)).
    # So at depth 2, states are (w2, w1) where w2 != s. So (s, w1) is not generated at depth 2.
    # Good.
    for s in range(n):
        # BFS in non-backtracking state space
        # state: (cur, prev), distance from s
        visited = set()
        q = deque()
        for v in adj[s]:
            visited.add((v, s))
            q.append((v, s, 1))
        while q:
            cur, prev, dist = q.popleft()
            if dist >= g - 1:
                continue
            # Check if we've returned to s (but only if dist < g, which is guaranteed by loop condition dist <= g-2, so dist+1 <= g-1 < g? No, dist is the length so far. If we extend to s, length becomes dist+1. We need dist+1 < g => dist < g-1.)
            # Actually, we check when we *generate* a neighbor.
            for nxt in adj[cur]:
                if nxt == prev:
                    continue  # backtracking
                new_state = (nxt, cur)
                if new_state in visited:
                    continue
                # If nxt == s, we found a cycle of length dist + 1
                if nxt == s:
                    if dist + 1 < g:
                        return False
                visited.add(new_state)
                q.append((nxt, cur, dist + 1))
    return True


def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar construction (baseline)."""
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


def _random_k_regular(n, k, seed=0):
    """Generate a random k-regular graph on n vertices using configuration model + dedup."""
    if n * k % 2 != 0:
        return None
    rng = random.Random(seed)
    # Configuration model
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    while True:
        rng.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        if valid and len(edges) == n * k // 2:
            return list(edges)
    return None


def _try_improve(n, k, g, edges, deadline, seed=0):
    """Try to reduce n by finding a smaller valid graph via random search."""
    # Moore bound for k-regular graph of girth g:
    # For g even: n >= 2 * (k-1)^(g/2 - 1) + k * (k-1)^(g/2 - 1) ... 
    # Actually, Moore bound: n >= k * (k-1)^{(g-2)/2} for odd g, n >= 2(k-1)^{g/2 - 1} for even g.
    # For k=4, g=10: n >= 2 * 3^4 = 162. But known lower bound is 243.
    # We try random graphs of size n' < n.
    best_n, best_edges = n, edges
    moore = 2 * (k - 1) ** (g // 2 - 1) if g % 2 == 0 else k * (k - 1) ** ((g - 2) // 2)
    # Start just above Moore bound
    for n_try in range(max(moore, k * (k - 1) ** ((g - 2) // 2)), best_n):
        if time.time() > deadline:
            break
        # Try a few random graphs of this size
        for attempt in range(20):
            if time.time() > deadline:
                break
            cand_edges = _random_k_regular(n_try, k, seed=seed + n_try * 100 + attempt)
            if cand_edges is None:
                continue
            if _is_valid(k, g, n_try, cand_edges):
                return n_try, cand_edges
    return best_n, best_edges


def construct(k, g):
    deadline = time.time() + 110  # Safety margin before 120s limit
    # Start with LUW construction
    n, edges = _luw_graph(k, g)
    # Verify it's valid
    if not _is_valid(k, g, n, edges):
        # Fallback: try random
        for attempt in range(50):
            if time.time() > deadline:
                break
            cand = _random_k_regular(n, k, seed=attempt)
            if cand and _is_valid(k, g, n, cand):
                edges = cand
                break
    # Try to improve
    n, edges = _try_improve(n, k, g, edges, deadline)
    return n, edges

# EVOLVE-BLOCK-END
