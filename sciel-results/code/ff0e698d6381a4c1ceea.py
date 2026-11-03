import time
import random
from collections import deque

_DEADLINE = 115.0  # safety margin vs 120s wall-clock


def _sl2_11():
    """Construct SL(2,11) as a list of (a,b,c,d) tuples with det=1 mod 11."""
    p = 11
    elements = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # Need ad - bc = 1 mod p
                # d = (1 + bc) * a^{-1} mod p if a != 0
                # If a == 0, then -bc = 1, so bc = -1 = 10 mod 11
                # d can be anything if a=0? No: ad-bc = 0*d - bc = -bc = 1 => bc = -1
                # So if a=0, we need bc=10, and d is free? No, det = ad-bc = 0-bc = -bc = 1 => bc=-1.
                # d doesn't appear in det when a=0. So d is free?
                # Wait, SL(2,p) has order p(p^2-1) = 11*120 = 1320.
                # Let's just iterate all and check det.
                for d in range(p):
                    det = (a * d - b * c) % p
                    if det == 1:
                        elements.append((a, b, c, d))
    return elements


def _mat_mul(x, y):
    a1, b1, c1, d1 = x
    a2, b2, c2, d2 = y
    p = 11
    return ((a1 * a2 + b1 * c2) % p,
            (a1 * b2 + b1 * d2) % p,
            (c1 * a2 + d1 * c2) % p,
            (c1 * b2 + d1 * d2) % p)


def _mat_inv(x):
    a, b, c, d = x
    p = 11
    # det = 1, so inv = [[d, -b], [-c, a]] mod p
    return (d, (-b) % p, (-c) % p, a)


def _construct_sl2_cayley(k, g, deadline):
    """Construct a 5-regular Cayley graph on SL(2,11) with girth >= g."""
    if k != 5 or g != 9:
        return None
    
    t0 = time.time()
    elements = _sl2_11()
    n = len(elements)
    idx = {e: i for i, e in enumerate(elements)}
    
    # Identity
    ident = (1, 0, 0, 1)
    
    # We need 5 generators. For a Cayley graph to be undirected and simple,
    # we need the connection set S to satisfy:
    # 1. 1 not in S
    # 2. S = S^{-1} (symmetric)
    # 3. No element in S has order 2 (to avoid 4-cycles from s^2=1? No, s^2=1 gives 2-cycle in directed, but in undirected it's just an edge. Wait. If s has order 2, s^{-1}=s. So the edge is {g, gs}. This is fine. But if we have two different order-2 elements s, t, then g, gs, gts, gt. Cycle g-gs-gts-gt-g? No, g to gt is not an edge unless t is in S.
    # Actually, for girth >= 9, we need no cycles of length 3,4,5,6,7,8.
    # A cycle of length 2 corresponds to s = t^{-1} for distinct s,t in S? No, s=s^{-1} is order 2.
    # In an undirected graph, an edge is {g, gs}. If s has order 2, it's just one edge.
    # If s has order > 2, s and s^{-1} are distinct. The edge {g, gs} is the same as {gs, g}.
    # So we should pick S such that S is symmetric.
    # To have 5 edges per vertex, we need |S| = 5.
    # If all elements in S have order 2, then S is a set of 5 involutions.
    # If some have order > 2, we need pairs {s, s^{-1}}.
    # Let's try to find 5 involutions (order 2 elements) in SL(2,11) that generate a large girth.
    
    # Find all involutions in SL(2,11)
    involutions = []
    for e in elements:
        if e == ident:
            continue
        e2 = _mat_mul(e, e)
        if e2 == ident:
            involutions.append(e)
    
    # There are 1320 elements. Number of involutions in SL(2,p) for p=11?
    # Trace 0. a+d=0 => d=-a. a^2+bc=1.
    # Let's just filter.
    
    if len(involutions) < 5:
        return None
    
    # We need to choose 5 involutions s1..s5 such that the Cayley graph has girth >= 9.
    # This is equivalent to: no non-trivial word of length < 9 in {s1..s5} equals identity,
    # except for backtracking (si si^{-1} = si si since si is involution).
    # So we need to avoid:
    # Length 3: s1 s2 s1 = 1 => s2 s1 s2 = s1? No. s1 s2 s1 = 1 => s1 s2 = s1^{-1} = s1 => s2=1. Impossible.
    # Wait, if s1, s2 are involutions, s1 s2 s1 = 1 implies s1 s2 = s1, so s2=1.
    # So no 3-cycles if all generators are involutions?
    # Let's check. A 3-cycle in Cayley graph with generators S is g, g s1, g s1 s2, g s1 s2 s3 = g.
    # So s1 s2 s3 = 1.
    # If s1,s2,s3 are involutions, s1 s2 s3 = 1 => s1 s2 = s3.
    # So we need to avoid s1 s2 = s3 for distinct s1,s2,s3 in S.
    
    # Length 4: s1 s2 s3 s4 = 1.
    # With involutions: s1 s2 s3 s4 = 1 => s1 s2 = s3 s4.
    # So we need to avoid s1 s2 = s3 s4 for distinct pairs.
    # Note s1 s2 = s2 s1 is not generally true.
    # Also s1 s2 = s1 s2 is trivial (backtracking? No, s1 s2 s1 s2 = (s1 s2)^2 = 1 is a 4-cycle if s1 s2 has order 2).
    # If s1 s2 has order 2, then s1 s2 s1 s2 = 1. This is a 4-cycle: g, g s1, g s1 s2, g s1 s2 s1, g.
    # So we need s1 s2 to have order > 2 for all distinct s1, s2 in S.
    
    # Let's try random sets of 5 involutions.
    
    rng = random.Random(42)
    
    # Precompute products of pairs of involutions to check order
    # And check s1 s2 = s3 for 3-cycles
    
    inv_set = set(involutions)
    
    # Check condition: for any distinct s1, s2 in S, s1 s2 is not identity (always true) and not an involution (order 2).
    # And for any distinct s1, s2, s3 in S, s1 s2 != s3.
    
    def check_set(S):
        S_list = list(S)
        # Check 3-cycles: s1 s2 = s3
        for i in range(len(S_list)):
            for j in range(len(S_list)):
                if i == j:
                    continue
                prod = _mat_mul(S_list[i], S_list[j])
                if prod in inv_set:
                    # Check if prod is in S and distinct from s1, s2?
                    # s1 s2 = s3. If s3 is in S, we have a 3-cycle.
                    # The cycle is g, g s1, g s1 s2 = g s3, g s3 s3 = g.
                    # Vertices: g, g s1, g s3.
                    # Edges: (g, g s1) via s1. (g s1, g s3) via s2? g s1 s2 = g s3. Yes.
                    # (g s3, g) via s3? g s3 s3 = g. Yes.
                    # So if s1 s2 = s3 for distinct s1,s2,s3 in S, we have a 3-cycle.
                    # But s1 s2 = s3 implies s3 is an involution.
                    # Is s3 distinct from s1 and s2?
                    # If s3 = s1, then s1 s2 = s1 => s2 = 1. Impossible.
                    # If s3 = s2, then s1 s2 = s2 => s1 = 1. Impossible.
                    # So s3 is distinct.
                    if prod in S:
                        return False
        # Check 4-cycles: (s1 s2)^2 = 1 => s1 s2 is involution.
        for i in range(len(S_list)):
            for j in range(len(S_list)):
                if i == j:
                    continue
                prod = _mat_mul(S_list[i], S_list[j])
                # Check if prod is involution
                prod2 = _mat_mul(prod, prod)
                if prod2 == ident:
                    # s1 s2 has order 2.
                    # Cycle: g, g s1, g s1 s2, g s1 s2 s1, g.
                    # Vertices: g, g s1, g s1 s2, g s2. (Since s1 s2 s1 = s2? No. s1 s2 s1 is not necessarily s2.)
                    # Wait. s1 s2 s1 s2 = 1 => s1 s2 s1 = s2^{-1} = s2.
                    # So g s1 s2 s1 = g s2.
                    # Vertices: g, g s1, g s1 s2, g s2.
                    # Edges: (g, g s1) via s1. (g s1, g s1 s2) via s2. (g s1 s2, g s2) via s1? g s1 s2 s1 = g s2. Yes.
                    # (g s2, g) via s2.
                    # This is a 4-cycle.
                    return False
        return True
    
    # Try random subsets
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        S = set(rng.sample(involutions, 5))
        if check_set(S):
            # Build graph
            S_list = list(S)
            adj = [[] for _ in range(n)]
            for u in range(n):
                for s in S_list:
                    v = _mat_mul(elements[u], s)
                    adj[u].append(idx[v])
            
            # Verify girth
            if _girth_bfs(n, adj, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    return None


def _girth_bfs(n, adj, limit):
    """Exact girth via non-backtracking BFS from every vertex. Returns True if
    girth >= limit."""
    for s in range(n):
        dist = {s: 0}
        prev = {s: -1}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u]
            if du + 1 >= limit:
                continue
            for v in adj[u]:
                if v == prev.get(u, -1):
                    continue
                if v not in dist:
                    dist[v] = du + 1
                    prev[v] = u
                    dq.append(v)
                else:
                    if du + 1 + dist[v] < limit:
                        return False
    return True


def _bipartite_luw(q, d):
    """Incidence graph of Lazebnik-Ustimenko-Woldar CD(d,q)."""
    q = int(q)
    d = int(d)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
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


def _has_short_cycle_3(n, a, b, c):
    """Check for cycles of length 3 in Cayley graph Z_n with generators {a, -a, b, -b, c}."""
    # a + b + c = 0 => a + b = n/2
    if (a + b) % n == c or (a - b) % n == c or (-a + b) % n == c or (-a - b) % n == c:
        return True
    # 2a + b = 0 => 2a = -b
    if (2 * a + b) % n == 0 or (2 * a - b) % n == 0:
        return True
    # a + 2b = 0 => a = -2b
    if (a + 2 * b) % n == 0 or (a - 2 * b) % n == 0:
        return True
    # 2a + c = 0 => 2a = -c = n/2
    if (2 * a + c) % n == 0 or (2 * a - c) % n == 0:
        return True
    # 2b + c = 0 => 2b = -c = n/2
    if (2 * b + c) % n == 0 or (2 * b - c) % n == 0:
        return True
    # 3a = 0
    if (3 * a) % n == 0:
        return True
    # 3b = 0
    if (3 * b) % n == 0:
        return True
    # 3c = 0 (c = n/2, 3n/2 != 0 mod n for even n)
    return False

def _has_short_cycle_4(n, a, b, c):
    """Check for non-backtracking cycles of length 4."""
    # a + a - a - a = 0 (backtracking, ignored)
    # a + b - a - b = 0 (backtracking, ignored)
    # a + a + b - b = 0 => 2a = 0
    if (2 * a) % n == 0:
        return True
    # b + b + a - a = 0 => 2b = 0
    if (2 * b) % n == 0:
        return True
    # a + b + c - c = 0 => a + b = 0 => a = -b
    if (a + b) % n == 0:
        return True
    # a + c + b - c = 0 => a + b = 0
    # a + c + c - a = 0 => 2c = 0 => n = 0
    # a + a + c - c = 0 => 2a = 0
    # a + b + a - b = 0 => 2a = 0
    # a + b + b - a = 0 => 2b = 0
    # a + c + b - b = 0 => a + c = 0 => a = -c = n/2
    if (a + c) % n == 0:
        return True
    # b + c + a - a = 0 => b + c = 0 => b = -c = n/2
    if (b + c) % n == 0:
        return True
    return False

def _search_cayley(k, g, max_n, deadline):
    """Search for a small k-regular Cayley graph on Z_n with girth >= g."""
    if k != 5:
        return None
    
    rng = random.Random(123)
    # Moore bound for k=5, g=9 is 426
    # We search n in [426, max_n)
    # n must be even for 5-regular graph on Z_n with symmetric generators?
    # 5n must be even => n even.
    
    for n in range(426, max_n, 2):
        if time.time() > deadline - 5:
            break
        c = n // 2
        # Try more random pairs
        for _ in range(100):
            a = rng.randint(1, n - 1)
            b = rng.randint(1, n - 1)
            if a == b or a == n - b or b == n - a or a == c or b == c:
                continue
            if _has_short_cycle_3(n, a, b, c):
                continue
            if _has_short_cycle_4(n, a, b, c):
                continue
            
            adj = [[] for _ in range(n)]
            for u in range(n):
                adj[u].extend([(u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n, (u + c) % n])
            
            if _girth_bfs(n, adj, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    return None


def construct(k, g):
    t0 = time.time()
    deadline = t0 + _DEADLINE

    # Baseline: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n_luw, edges_luw = _bipartite_luw(q, d)
    
    best_n, best_edges = n_luw, edges_luw
    
    # Try to find a smaller Cayley graph on Z_n
    if k == 5 and g == 9:
        result = _search_cayley(k, g, best_n, deadline)
        if result:
            n_cay, edges_cay = result
            if n_cay < best_n:
                best_n, best_edges = n_cay, edges_cay
        
        # Try SL(2,11) Cayley graph
        result_sl2 = _construct_sl2_cayley(k, g, deadline)
        if result_sl2:
            n_sl2, edges_sl2 = result_sl2
            if n_sl2 < best_n:
                best_n, best_edges = n_sl2, edges_sl2

    return best_n, best_edges
