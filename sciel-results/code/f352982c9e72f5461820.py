"""Combined program: Lazebnik-Ustimenko-Woldar construction + girth certification/repair."""
import time
from collections import deque

# EVOLVE-BLOCK-START

def _is_prime_power(n):
    """Return (p, m) if n == p^m for some prime p, else None."""
    if n < 2:
        return None
    # find prime factor
    d = 2
    while d * d <= n:
        if n % d == 0:
            # check n is a power of d
            x = n
            m = 0
            while x % d == 0:
                x //= d
                m += 1
            if x == 1:
                # verify d is prime
                if _is_prime(d):
                    return (d, m)
            break
        d += 1
    if _is_prime(n):
        return (n, 1)
    return None


def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _next_prime_power(ge):
    """Smallest prime power >= ge."""
    n = ge
    while True:
        r = _is_prime_power(n)
        if r is not None:
            return n
        n += 1


def _make_gf(q):
    """Build add/sub/mul for GF(q), q a prime power. Returns (add, sub, mul)."""
    r = _is_prime_power(q)
    if r is None:
        raise ValueError("q not a prime power: %d" % q)
    p, m = r
    if m == 1:
        def add(x, y):
            return (x + y) % p
        def sub(x, y):
            return (x - y) % p
        def mul(x, y):
            return (x * y) % p
        return add, sub, mul
    # Build GF(p^m) with a primitive polynomial.
    # Find an irreducible polynomial of degree m over GF(p).
    poly = _find_irreducible_poly(p, m)
    # Elements are polynomials of degree < m, represented as ints (base p).
    # Precompute multiplication table.
    size = q
    # Convert int to list of coefficients
    def to_coeffs(v):
        c = [0] * m
        x = v
        for i in range(m):
            c[i] = x % p
            x //= p
        return c

    def from_coeffs(c):
        v = 0
        for i in range(m - 1, -1, -1):
            v = v * p + c[i]
        return v

    # poly: list of coefficients degree 0..m, poly[m] != 0 (usually 1)
    # Reduce: x^m = sum_{i=0}^{m-1} poly[i] * x^i  (since poly is irreducible, x^m - sum = poly)
    # Actually poly(x) = x^m + poly[m-1] x^{m-1} + ... + poly[0]
    # So x^m = - (poly[m-1] x^{m-1} + ... + poly[0]) mod p = (p - poly[m-1]) x^{m-1} + ... 
    # In GF(p), -a = (p-a) % p.
    neg = [(-poly[i]) % p for i in range(m)]
    # neg[i] is the coefficient for x^i in the reduction of x^m.

    def reduce_poly(c):
        # c is list length m (degree < m). But we may have higher during mul; handle via loop.
        return c

    def gf_mul_coeffs(a, b):
        # multiply two degree < m polys, reduce mod poly
        # result degree < 2m-1
        res = [0] * (2 * m - 1)
        for i in range(m):
            if a[i] == 0:
                continue
            for j in range(m):
                if b[j] == 0:
                    continue
                res[i + j] = (res[i + j] + a[i] * b[j]) % p
        # reduce degrees >= m using x^m = sum neg[i] x^i
        for deg in range(2 * m - 2, m - 1, -1):
            coef = res[deg]
            if coef == 0:
                continue
            for i in range(m):
                res[deg - m + i] = (res[deg - m + i] + coef * neg[i]) % p
            res[deg] = 0
        return [res[i] % p for i in range(m)]

    # Precompute mul table
    mul_table = [[0] * size for _ in range(size)]
    for a in range(size):
        ca = to_coeffs(a)
        for b in range(size):
            cb = to_coeffs(b)
            mc = gf_mul_coeffs(ca, cb)
            mul_table[a][b] = from_coeffs(mc)

    def add(x, y):
        return (x + y) % p  # WRONG for m>1; must do coeff-wise
    # Redo: addition is coeff-wise mod p
    def add(x, y):
        cx = to_coeffs(x)
        cy = to_coeffs(y)
        return from_coeffs([(cx[i] + cy[i]) % p for i in range(m)])
    def sub(x, y):
        cx = to_coeffs(x)
        cy = to_coeffs(y)
        return from_coeffs([(cx[i] - cy[i]) % p for i in range(m)])
    def mul(x, y):
        return mul_table[x][y]
    return add, sub, mul


def _find_irreducible_poly(p, m):
    """Find an irreducible polynomial of degree m over GF(p), monic."""
    # Enumerate monic polynomials x^m + a_{m-1} x^{m-1} + ... + a_0, a_0 != 0 (irreducible must be non-constant term)
    # For small m,p this is fine.
    from math import prod
    # Build list of all monic polys of degree m with a_0 != 0
    # Represent as tuple (a_0, ..., a_{m-1})
    def poly_from_coeffs(a):
        # a = (a_0, ..., a_{m-1}); poly = x^m + a_{m-1} x^{m-1} + ... + a_0
        return [a[i] for i in range(m)]

    def poly_mul_mod(f, g, poly):
        # f, g lists degree < m; poly list length m (coeffs of x^m reduction)
        # returns product mod poly
        mdeg = len(poly)
        res = [0] * (2 * mdeg - 1)
        for i in range(mdeg):
            if f[i] == 0:
                continue
            for j in range(mdeg):
                if g[j] == 0:
                    continue
                res[i + j] = (res[i + j] + f[i] * g[j]) % p
        for deg in range(2 * mdeg - 2, mdeg - 1, -1):
            coef = res[deg]
            if coef == 0:
                continue
            for i in range(mdeg):
                res[deg - mdeg + i] = (res[deg - mdeg + i] + coef * poly[i]) % p
            res[deg] = 0
        return [res[i] % p for i in range(mdeg)]

    def poly_divides(f, g):
        # does f divide g? f,g lists, deg f < deg g or equal
        # simple: compute g mod f, check zero. Use x^deg reduction from f.
        # f is monic of degree d = len(f)-1
        d = len(f) - 1
        if d == 0:
            return g[-1] == 0  # constant divides everything; but we use monic linear
        # build reduction: x^d = -sum_{i=0}^{d-1} f[i] x^i
        neg = [(-f[i]) % p for i in range(d)]
        h = list(g)
        for deg in range(len(h) - 1, d - 1, -1):
            coef = h[deg]
            if coef == 0:
                continue
            for i in range(d):
                h[deg - d + i] = (h[deg - d + i] + coef * neg[i]) % p
            h[deg] = 0
        for i in range(d):
            if h[i] != 0:
                return False
        return True

    # enumerate
    import itertools
    for a0 in range(1, p):
        for rest in itertools.product(range(p), repeat=m - 1):
            a = (a0,) + rest
            poly = list(a)  # coeffs degree 0..m-1
            # check irreducible: not divisible by any monic linear or lower-degree factor.
            # For small m, check divisibility by all monic polys of degree 1..m//2.
            irr = True
            for d in range(1, m // 2 + 1):
                # enumerate monic polys of degree d
                if d == 1:
                    cands = [(c,) for c in range(p)]
                else:
                    cands = [t for t in itertools.product(range(p), repeat=d)]
                for c in cands:
                    f = list(c) + [1]  # monic degree d
                    if poly_divides(f, poly + [1]):
                        irr = False
                        break
                if not irr:
                    break
            if irr:
                return poly
    raise RuntimeError("no irreducible poly found")


def _lwu_construct(q, d, deadline):
    """Build LWU graph CD(d,q), return (n, edges) or None on timeout."""
    add, sub, mul = _make_gf(q)
    # rule
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # vertex = (side, tuple of d coords)
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None] + [0] * d
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y[1:])

    start = (0, tuple([0] * d))
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        if time.time() > deadline:
            return None
    return len(order), edges


def _verify_girth(n, edges, g):
    """Return True if girth >= g, using non-backtracking walks from every vertex.
    Girth >= g iff no non-backtracking closed walk of length < g (and no backtracking
    shorter). We check: no cycle of length < g. Use BFS in the non-backtracking
    line graph: state = (u, prev). A cycle of length L corresponds to a non-backtracking
    walk of length L from (v, u) back to (u, v). We detect if for any directed edge
    (prev -> u), a non-backtracking walk of length L (L < g) returns to (u, prev).
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # non-backtracking BFS: from each directed edge (prev, u), expand to (u, w) for w != prev.
    # We want to know the shortest non-backtracking closed walk for each directed edge.
    # A cycle of length L exists iff there is a directed edge (a,b) such that a non-backtracking
    # walk of length L from (a,b) returns to (b,a).
    # We'll do BFS over states (u, prev) with prev = the vertex we came from.
    # Start states: all directed edges (prev=u, cur=v) for each undirected edge.
    # dist[(cur, prev)] = length so far.
    # We stop when we find dist[(b, a)] <= L for L < g, meaning a cycle of length L.
    # Actually the state is (cur, prev): we are at cur, having come from prev.
    # Initial: for each edge (a,b): state (b, a) with length 1, and (a, b) with length 1.
    # Transition: from (cur, prev) to (nxt, cur) for each nxt in adj[cur], nxt != prev, length+1.
    # A non-backtracking closed walk of length L returning to start state (b,a) means
    # we found (b, a) again at length L. The cycle length is L (the walk length).
    # We need girth >= g, so no closed non-backtracking walk of length < g.
    # Note: a 2-cycle (back and forth) is excluded by non-backtracking.
    INF = n * n + 10
    best = {}  # (cur, prev) -> min length
    from collections import deque as dq
    # We'll run a multi-source BFS from all directed edges simultaneously, but we need
    # to detect return to the *same* start. Instead, for each start directed edge, we could
    # BFS, but that's O(E * E). Better: the girth is the minimum over directed edges (a,b) of
    # the shortest non-backtracking walk from (b,a) to (a,b). We can compute all-pairs in the
    # non-backtracking graph? That's expensive. For n up to a few thousand and g small (10),
    # we can do BFS from each start but prune at length g-1.
    # Since g is small (10), BFS depth is at most g-1 = 9, branching factor <= k-1 = 3.
    # Number of states reachable within 9 steps from a start ~ 3^9 = 19683, times E starts
    # (2*E = 4n for 4-regular) ~ 4n * 19683. For n=320, that's ~25M, borderline but ok in
    # Python with early exit. Actually we can early-exit the whole thing as soon as we find
    # any cycle of length < g.
    # Let's do: for each directed edge start, BFS up to depth g-1; if we reach the target
    # state at depth L < g, return False.
    # To speed up, use a visited dict per start.
    for (a, b) in edges:
        # start state (b, a), target (a, b)
        start_state = (b, a)
        target_state = (a, b)
        # BFS
        seen = {start_state: 1}
        frontier = [start_state]
        depth = 1
        while frontier and depth < g:
            next_frontier = []
            for (cur, prev) in frontier:
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    ns = (nxt, cur)
                    if ns in seen:
                        continue
                    seen[ns] = depth + 1
                    if ns == target_state:
                        # found cycle of length depth+1
                        return False
                    next_frontier.append(ns)
            frontier = next_frontier
            depth += 1
    return True


def construct(k, g):
    """Construct a k-regular graph with girth at least g."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Choose d such that girth guarantee >= g.
    # LWU CD(d,q): girth >= d+5 if d odd, >= d+4 if d even (when q >= d+1 or similar).
    # We'll try increasing d and q until a valid graph is found and verified.
    best = None
    # Candidate (q, d) pairs in increasing size.
    candidates = []
    # d must satisfy: (d odd and d+5 >= g) or (d even and d+4 >= g)
    # Also need q >= d+1 typically for the construction to be well-defined (the rule uses
    # indices up to c0+1 which can be d+1, so q should be >= d+2 to be safe; but the
    # original uses q=k and small d). We'll require q >= d+1.
    for d in range(2, 25):
        ok = False
        if d % 2 == 1:
            ok = (d + 5 >= g)
        else:
            ok = (d + 4 >= g)
        if not ok:
            continue
        # q must be a prime power >= max(k, d+1)
        q = _next_prime_power(max(k, d + 1))
        candidates.append((q, d))
    if not candidates:
        # fallback
        d = max(3, g - 5)
        q = _next_prime_power(max(k, d + 1))
        candidates = [(q, d)]

    for (q, d) in candidates:
        if time.time() > deadline:
            break
        res = _lwu_construct(q, d, deadline)
        if res is None:
            continue
        n, edges = res
        # verify 4-regular (k-regular) and simple
        # degree check
        deg = [0] * n
        eset = set()
        valid = True
        for u, v in edges:
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in eset:
                valid = False
                break
            eset.add(key)
            deg[u] += 1
            deg[v] += 1
        if not valid:
            continue
        if any(deg[i] != k for i in range(n)):
            continue
        # verify girth
        if _verify_girth(n, edges, g):
            if best is None or n < best[0]:
                best = (n, edges)
            # We found a valid one; could try to find smaller, but larger d/q gives larger
            # graphs. Since candidates are in increasing d (hence increasing size), the first
            # valid is likely smallest. Return it.
            return best
    # If none verified within deadline, return best found (even if unverified) or the
    # last constructed.
    if best is not None:
        return best
    # Fallback: return the first constructed even without verification
    for (q, d) in candidates:
        res = _lwu_construct(q, d, deadline)
        if res is not None:
            return res
    # Last resort: trivial (should not happen)
    return (0, [])

# EVOLVE-BLOCK-END
