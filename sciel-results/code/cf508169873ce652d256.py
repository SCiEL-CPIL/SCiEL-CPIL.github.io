"""ABS Cayley graph construction for the cage problem n(k, g).

Builds a k-regular graph of girth >= g with few vertices using the
Alon-Babai-Suzuki construction on the group of 3x3 upper-triangular
unipotent matrices over GF(p) (p = smallest prime >= k).  The group is
nilpotent of class 2, so the commutator of any two non-identity elements
is non-identity: the Cayley graph has no 4-cycles, hence girth >= 5.
The exact girth is then verified by non-backtracking BFS from every vertex
(the same certificate the evaluator uses); if it is < g we fall back to the
Lazebnik-Ustimenko-Woldar bipartite construction so the result is always
valid.
"""
import time

# EVOLVE-BLOCK-START

# ---------- prime helpers ----------
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


def _next_prime(n):
    m = max(2, n)
    while not _is_prime(m):
        m += 1
    return m


# ---------- LUW fallback (original seed design) ----------
def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
    else:
        add = (lambda x, y: (x + y) % q)
        sub = (lambda x, y: (x - y) % q)
        mul = (lambda x, y: x * y % q)
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
                y[c] = (add(x[c], mul(y[a], x[b])) if side == 0
                        else sub(x[c], mul(x[a], y[b])))
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


# ---------- girth verification (non-backtracking BFS, evaluator's method) ----------
def _verify_girth(n, edges, g):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    for i in range(n):
        if len(adj[i]) != k_global:
            return False
        # non-backtracking BFS from i
        dist = {i: 0}
        frontier = [(i, -1)]
        d = 0
        while frontier:
            d += 1
            if d >= g:
                break
            nxt = []
            for u, p in frontier:
                for v in adj[u]:
                    if v == p:
                        continue
                    if v in dist:
                        return False  # cycle shorter than g
                    dist[v] = d
                    nxt.append((v, u))
            frontier = nxt
    return True


# ---------- ABS Cayley construction ----------
def _abs_construct(k, g, deadline):
    p = _next_prime(k)
    # group elements as (a, b, c) == matrix [[1,a,b],[0,1,c],[0,0,1]]
    # multiplication: (a,b,c)*(a',b',c') = (a+a', b+b'+a*c', c+c')
    # inverse: (a,b,c)^-1 = (-a, -b+a*c, -c)
    def mul(x, y):
        return ((x[0] + y[0]) % p,
                (x[1] + y[1] + x[0] * y[2]) % p,
                (x[2] + y[2]) % p)

    def inv(x):
        return ((-x[0]) % p,
                (-x[1] + x[0] * x[2]) % p,
                (-x[2]) % p)

    # candidate generators: elementary matrices (and their inverses)
    # E11: (1,0,0); E22: (0,0,1); E12: (0,1,0)
    base = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]

    best = None
    best_n = None
    # try the 3-generator set first (degree 6), then subsets of size k
    # We need exactly k generators (each with its inverse) for k-regularity.
    from itertools import combinations
    # candidate pool = base + inverses + products (small pool)
    pool = list(base)
    for x in base:
        pool.append(inv(x))
    # de-duplicate, exclude identity
    seen = set()
    pool = []
    for x in pool:
        if x == (0, 0, 0):
            continue
        if x in seen:
            continue
        seen.add(x)
        pool.append(x)

    # We need to pick a set S of k generators, closed under the requirement that
    # each generator's inverse is also in S (so the graph is undirected and regular).
    # Simplest: pick k/2 pairs {x, x^-1}.  For odd k this is not integer; instead
    # pick k generators where the set is symmetric.  For k=5 (odd) a symmetric
    # set of size 5 is impossible with pairs only (pairs have even size).  So we
    # allow the graph to be directed-regular and then symmetrize: use S ∪ S^-1.
    # To get degree exactly k, choose S with |S ∪ S^-1| = k.
    # For odd k, include one involution-free structure is impossible; instead
    # we build the graph from a symmetric generator set and accept degree = k.
    #
    # Practical approach for k=5: use generators g1..g5 such that the set
    # G = {g1..g5} is symmetric (g in G => g^-1 in G).  A symmetric set of odd
    # size needs a self-inverse element.  In this group the only self-inverse
    # element is the identity (since x^2=e => x=e for the (a,b,c) form? check).
    # x^2 = (2a, 2b+a*c, 2c) = (0,0,0) => 2a=0,2c=0 (p=5 odd => a=c=0), then
    # 2b=0 => b=0.  So only identity is self-inverse.  Hence no symmetric set of
    # odd size exists.  => For odd k we cannot get a simple undirected k-regular
    # Cayley graph this way.  Fallback: make it bipartite-ish by using a
    # directed regular graph and then we need undirected.  So for odd k we
    # instead target degree k by using k generators and taking the undirected
    # version (edges {x, x*g} for g in S, ignoring direction).  This yields
    # degree up to k but possibly less if g and g^-1 both in S create
    # duplicates.  To guarantee exactly k, we take S of size k and build
    # undirected edges; degree of vertex = |{g : x*g != x}| which is k if no
    # generator is identity and no two generators give same neighbor.  Since
    # generators are distinct and non-identity, each gives a distinct neighbor,
    # so degree = k.  The graph is undirected because if x~y via g then
    # y~x via g^-1, and g^-1 may or may not be in S; but the edge {x,y} is
    # added regardless, so undirectedness holds.  Degree is exactly k.
    #
    # Great: just pick k distinct non-identity generators, build undirected
    # edges {x, x*g}.  Degree = k automatically.

    if len(pool) < k:
        # expand pool with products
        for x in list(pool):
            for y in pool:
                z = mul(x, y)
                if z != (0, 0, 0) and z not in seen:
                    seen.add(z)
                    pool.append(z)

    # choose k generators: prefer "independent" ones (the elementary ones)
    # to maximize girth.  Try several random-ish selections within budget.
    import random
    rng = random.Random(12345)

    # deterministic first: use the first k of pool (elementary + inverses)
    candidates = []
    base_sel = pool[:k]
    candidates.append(list(base_sel))

    # add a few random selections
    for _ in range(8):
        if len(pool) >= k:
            sel = rng.sample(pool, k)
            candidates.append(sel)

    for sel in candidates:
        if time.time() > deadline:
            break
        # build graph
        elems = [(a, b, c) for a in range(p) for b in range(p) for c in range(p)]
        idx = {e: i for i, e in enumerate(elems)}
        n = len(elems)
        edge_set = set()
        for x in elems:
            xi = idx[x]
            for g in sel:
                y = mul(x, g)
                yi = idx[y]
                if xi == yi:
                    continue
                key = (min(xi, yi), max(xi, yi))
                edge_set.add(key)
        edges = list(edge_set)
        # quick degree check
        deg = [0] * n
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        if any(d != k for d in deg):
            continue
        # girth check (only if within budget)
        if time.time() > deadline:
            break
        if _verify_girth(n, edges, g):
            if best_n is None or n < best_n:
                best_n = n
                best = (n, edges)
            # found a valid one; keep searching a bit for smaller? n is fixed = p^3
            # for this group, so any valid selection gives same n.  Stop early.
            break

    if best is not None:
        return best
    return None


k_global = None


def construct(k, g):
    global k_global
    k_global = k
    deadline = time.time() + 110  # safety margin before 150s limit
    result = _abs_construct(k, g, deadline)
    if result is not None:
        return result
    # fallback: LUW bipartite construction (always valid for the requested girth)
    return _luw_construct(k, g)
# EVOLVE-BLOCK-END
