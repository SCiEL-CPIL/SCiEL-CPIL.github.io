"""Construct small 4-regular graphs of high girth via algebraic constructions."""
import time
import math

# Deadline guard: evaluator kills at 150s; keep a safety margin.
_DEADLINE = 145.0


def _now():
    return time.time()


def _girth_check(n, edges, g):
    """Return True if the graph has no cycle shorter than g (using BFS from each vertex)."""
    from collections import deque
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # BFS from each vertex, tracking distance; if we find a shorter cycle return False.
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= g:  # no need to go deeper than g
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif dist[v] <= dist[u]:
                    # cycle length = dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True


def _verify_regular(n, edges, k):
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return all(d == k for d in deg)


def _verify_simple(n, edges):
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
    return True


def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges).
    """
    start_time = _now()
    deadline = start_time + _DEADLINE

    # For the specific case k=4, g=10, we can use known algebraic constructions.
    # The smallest known 4-regular graph of girth 10 has 320 vertices.
    # Let's try several constructions and pick the best valid one.

    candidates = []

    # Construction 1: Lazebnik-Ustimenko-Woldar (L-U-W) graph
    # The seed program uses this. Let's keep it as a fallback.
    try:
        n_luw, edges_luw = _luw_construction(k, g)
        candidates.append((n_luw, edges_luw))
    except Exception:
        pass

    # Construction 2: Generalized polygon incidence graph
    # A generalized (s,t)-gon has incidence graph that is (s+1)-regular with girth 4*ceil((s+1+t+1)/2) ... 
    # Actually, the incidence graph of a generalized h-gon of order (s,t) is (s+1)-regular 
    # and has girth 2h where h is the number of sides.
    # For a generalized hexagon (h=6), girth = 12, which satisfies g=10.
    # For a generalized pentagon (h=5), girth = 10, which satisfies g=10.
    # The smallest generalized pentagon is the Hall-Janko generalized pentagon with s=2, t=2, 
    # giving a 3-regular graph of girth 10 with 112 vertices. But we need 4-regular.
    # The incidence graph of a generalized pentagon of order (s,t) is (s+1)-regular.
    # So we need s+1 = 4, i.e., s=3.
    # The smallest generalized pentagon of order (3, t) has (s+1)(t+1) * 5 / 2 vertices? 
    # Actually, the number of points = (s+1)(s^2+t^2+st+1) ... this gets complicated.
    # Let's use the known fact: the incidence graph of the generalized pentagon of order (3,3) 
    # is 4-regular with girth 10. The number of vertices is 2 * (number of points) where 
    # number of points = (s+1)(s^2 + st + t^2 + 1) ... 
    # Actually for a generalized pentagon of order (s,t): 
    #   #points = (s+1)(s^2 + st + t^2 + 1)
    #   #lines = (t+1)(s^2 + st + t^2 + 1)
    # For s=t=3: #points = 4*(9+9+9+1) = 4*28 = 112, #lines = 112. Total vertices = 224.
    # This gives a 4-regular graph of girth 10 with 224 vertices!
    try:
        n_gp, edges_gp = _generalized_pentagon_incidence(3, 3)
        candidates.append((n_gp, edges_gp))
    except Exception:
        pass

    # Construction 3: Moore bound check
    # The Moore bound for k=4, g=10 (even girth) is:
    # n >= 1 + k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^{g/2 - 1})
    # = 1 + 4 * (1 + 3 + 9 + 27 + 81) = 1 + 4 * 121 = 1 + 484 = 485? 
    # Wait, that's for trees. The Moore bound for girth g (even) is:
    # n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i = 2 * (1 + 3 + 9 + 27 + 81) = 2 * 121 = 242
    # So the lower bound is 242. The best known is 320.
    # Our generalized pentagon construction gives 224, which is below the Moore bound?
    # That can't be right. Let me recheck.
    # The Moore bound for a k-regular graph of girth g (even) is:
    # n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^{g/2 - 1})
    # For k=4, g=10: g/2 = 5, so sum i=0 to 4: 1 + 3 + 9 + 27 + 81 = 121. n >= 2*121 = 242.
    # So 224 < 242, which means either my formula for the generalized pentagon is wrong,
    # or the girth is not actually 10, or the graph is not 4-regular.
    # Let me reconsider. The incidence graph of a generalized pentagon of order (s,t) 
    # is (s+1)-regular only if s=t. If s != t, it's not regular.
    # For s=t=3, it is 4-regular. The girth is 2*5 = 10. 
    # The number of points in a generalized pentagon of order (s,s) is:
    # (s+1)(s^2 + s^2 + s^2 + 1) = (s+1)(3s^2 + 1)
    # For s=3: 4*(27+1) = 4*28 = 112 points, 112 lines, 224 total vertices.
    # But 224 < 242, which violates the Moore bound. So something is wrong.
    # Actually, I think the Moore bound I recalled is for the case where the graph is a tree 
    # up to radius g/2. Let me look this up more carefully.
    # The Moore bound M(k,g) for even g=2r is: n >= 2 * (1 + (k-1) + ... + (k-1)^{r-1})
    # For k=4, r=5: n >= 2*(1+3+9+27+81) = 2*121 = 242.
    # So a 4-regular graph of girth 10 must have at least 242 vertices.
    # Therefore, my generalized pentagon construction must be incorrect.
    # Let me just use the L-U-W construction and a few other known constructions.

    # Let's also try the known 320-vertex construction if available.
    # For now, let's just use the L-U-W construction which is valid.

    # Pick the best candidate
    if not candidates:
        # Fallback: return a trivial valid graph if possible
        raise ValueError("No valid construction found")

    # Sort by number of vertices
    candidates.sort(key=lambda x: x[0])

    # Verify the best candidate
    best_n, best_edges = candidates[0]
    if _verify_regular(best_n, best_edges, k) and _verify_simple(best_n, best_edges):
        return best_n, best_edges

    # If verification fails, try the next candidate
    for n, edges in candidates[1:]:
        if _verify_regular(n, edges, k) and _verify_simple(n, edges):
            return n, edges

    raise ValueError("No valid construction found")


def _luw_construction(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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


def _generalized_pentagon_incidence(s, t):
    """
    Construct the incidence graph of a generalized pentagon of order (s,t).
    This is a placeholder; the actual construction is complex.
    For s=t=3, this would give a 4-regular graph of girth 10.
    """
    # This is a stub; the actual construction requires explicit coordinates.
    # For now, return an empty result to skip this candidate.
    raise NotImplementedError
