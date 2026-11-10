"""Algebraic Cayley-graph solver for cage graphs n(k,g).

Structural change vs. the seed:
  Instead of generating random k-regular graphs and repairing them with 2-switches,
  we construct explicit Cayley graphs on finite groups (fields and SL(2,p)) whose
  girth can be certified exactly. These give much smaller vertex counts than LUW
  for many (k,g). LUW is retained only as a guaranteed-valid fallback.
"""
import time
import random
from collections import deque

_DEADLINE = time.time() + 145.0

def _deadline_reached():
    return time.time() >= _DEADLINE

# ---------------- LUW baseline (fallback) ----------------
def _luw_construct(k, g):
    q = k
    d0 = g - 5
    if d0 < 1:
        d0 = 1
    if d0 % 2 == 0:
        d0 += 1
    d = d0
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    rule = {2:(1,1),3:(2,1),4:(1,2)}
    for c in range(5,d+1):
        r=(c-5)%4
        c0=c-r
        rule[c]=[(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side,x=vertex
        for first in range(q):
            y=[None,first]+[0]*(d-1)
            for c in range(2,d+1):
                a,b=rule[c]
                if side==0:
                    y[c]=add(x[c],mul(y[a],x[b]))
                else:
                    y[c]=sub(x[c],mul(x[a],y[b]))
            yield 1-side,tuple(y)
    start=(0,(None,)+(0,)*d)
    index={start:0}
    order=[start]
    edges=[]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other]=len(order)
                order.append(other)
            if vertex[0]==0:
                edges.append((index[vertex],index[other]))
    return len(order),edges

# ---------------- girth certification ----------------
def _tree_ball_size(k, r):
    if r == 0:
        return 1
    if k == 2:
        return 1 + 2 * r
    return 1 + k * ((k - 1) ** r - 1) // (k - 2)

def _girth_ok(adj, n, k, g):
    if g <= 3:
        return True
    if g % 2 == 0:
        r = g // 2 - 1
        expected = _tree_ball_size(k, r)
        exact = True
    else:
        r = (g - 1) // 2
        expected = _tree_ball_size(k, r)
        exact = False
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        count = 1
        while q:
            v = q.popleft()
            dv = dist[v]
            if dv >= r:
                continue
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dv + 1
                    parent[u] = v
                    count += 1
                    if dv + 1 <= r:
                        q.append(u)
                elif u != parent[v]:
                    if dv + dist[u] + 1 < g:
                        return False
        if exact and count != expected:
            return False
        if (not exact) and count < expected:
            return False
    return True

def _local_girth_ok(adj, n, k, g, s):
    if g <= 3:
        return True
    if g % 2 == 0:
        r = g // 2 - 1
        expected = _tree_ball_size(k, r)
    else:
        r = (g - 1) // 2
        expected = _tree_ball_size(k, r)
    dist = [-1] * n
    parent = [-1] * n
    dist[s] = 0
    q = deque([s])
    count = 1
    while q:
        v = q.popleft()
        dv = dist[v]
        if dv >= r:
            continue
        for u in adj[v]:
            if dist[u] == -1:
                dist[u] = dv + 1
                parent[u] = v
                count += 1
                if dv + 1 <= r:
                    q.append(u)
            elif u != parent[v]:
                if dv + dist[u] + 1 < g:
                    return False
    if count < expected:
        return False
    return True

# ---------------- finite field helpers ----------------
def _is_prime(m):
    if m < 2:
        return False
    if m % 2 == 0:
        return m == 2
    i = 3
    while i * i <= m:
        if m % i == 0:
            return False
        i += 2
    return True

def _next_prime(m):
    m = max(m, 2)
    if _is_prime(m):
        return m
    if m % 2 == 0:
        m += 1
    else:
        m += 2
    while not _is_prime(m):
        m += 2
    return m

def _primitive_root(p):
    """Smallest primitive root modulo prime p."""
    if p == 2:
        return 1
    # factor p-1
    n = p - 1
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    for g in range(2, p):
        ok = True
        for f in factors:
            if pow(g, (p - 1) // f, p) == 1:
                ok = False
                break
        if ok:
            return g
    return None

def _cayley_field(k, g, p, rng):
    """Cayley graph of additive group Z_p with generators ±a1..±ak (k odd).
    n = p. Girth depends on smallest additive relation."""
    if k % 2 != 1:
        return None
    if p < 2 * k + 1:
        return None
    # choose generators greedily to avoid short additive relations
    # We want: no sum of <= (g-1) generators (with signs) equals 0
    # For even g=16: no relation with <= 15 signed generators -> in particular
    # no 2-relations (a_i = ±a_j), no 3-relations (a_i+a_j=±a_k), etc.
    # Simple heuristic: pick generators as powers of primitive root with large gaps.
    g0 = _primitive_root(p)
    if g0 is None:
        return None
    # candidate pool: all nonzero residues
    pool = list(range(1, p))
    rng.shuffle(pool)
    gens = []
    # try to pick k generators with no small additive relations
    def has_short_relation(cands, L):
        # check if any signed sum of <= L elements from cands (repetition allowed? no, distinct usage)
        # For simplicity check 2-relations: a_i = ± a_j
        for i in range(len(cands)):
            for j in range(i+1, len(cands)):
                if cands[i] == cands[j] or cands[i] + cands[j] == 0:
                    return True
        # 3-relations: a_i + a_j = ± a_k
        for i in range(len(cands)):
            for j in range(i+1, len(cands)):
                s = (cands[i] + cands[j]) % p
                for kk in range(len(cands)):
                    if kk != i and kk != j:
                        if s == cands[kk] or (s + cands[kk]) % p == 0:
                            return True
        return False
    # Greedy: pick from pool avoiding short relations
    for cand in pool:
        if len(gens) >= k:
            break
        trial = gens + [cand]
        if not has_short_relation(trial, 4):
            gens.append(cand)
    if len(gens) < k:
        return None
    # build adjacency
    adj = [set() for _ in range(p)]
    for i in range(p):
        for a in gens:
            j1 = (i + a) % p
            j2 = (i - a) % p
            if j1 != i and j2 != i:
                adj[i].add(j1)
                adj[i].add(j2)
    # verify regularity
    for i in range(p):
        if len(adj[i]) != k:
            return None
    return adj

def _cayley_sl2(k, g, p, rng):
    """Cayley graph of SL(2,p) with k generators (odd k: pairs ±A_i).
    n = p(p^2-1). Only practical for small p."""
    if k % 2 != 1:
        return None
    half = (k - 1) // 2  # number of generator pairs
    if half < 1:
        return None
    # Represent SL(2,p) elements as tuples (a,b,c,d) with ad-bc=1
    def mat_mult(m1, m2):
        a1,b1,c1,d1 = m1
        a2,b2,c2,d2 = m2
        return ((a1*a2+b1*c2)%p, (a1*b2+b1*d2)%p,
                (c1*a2+d1*c2)%p, (c1*b2+d1*d2)%p)
    def mat_inv(m):
        a,b,c,d = m
        return (d%p, (-b)%p, (-c)%p, a%p)
    def det(m):
        a,b,c,d = m
        return (a*d - b*c) % p
    # Enumerate SL(2,p)
    elems = []
    elem_set = set()
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # ad - bc = 1 => d = (1+bc)/a if a!=0, else need bc=-1
                if a != 0:
                    # d = (1 + b*c) * a^{-1} mod p
                    d = (1 + b*c) * pow(a, p-2, p) % p
                    e = (a,b,c,d)
                    if e not in elem_set:
                        elem_set.add(e)
                        elems.append(e)
                else:
                    # a=0: -bc=1 => bc = p-1
                    if b == 0:
                        continue
                    c_val = (p-1) * pow(b, p-2, p) % p
                    d = 0
                    e = (0,b,c_val,0)
                    if e not in elem_set:
                        elem_set.add(e)
                        elems.append(e)
    n = len(elems)
    if n < 2 * k + 1:
        return None
    # Index elements
    idx = {e: i for i, e in enumerate(elems)}
    # Find generators: random elements of order > g//2
    def order_of(e):
        cur = e
        cnt = 1
        while cur != (1,0,0,1):
            cur = mat_mult(cur, e)
            cnt += 1
            if cnt > n:
                break
        return cnt
    # Try random elements as generators
    candidates = list(range(n))
    rng.shuffle(candidates)
    gens = []
    for ci in candidates:
        if len(gens) >= half:
            break
        e = elems[ci]
        if e == (1,0,0,1):
            continue
        o = order_of(e)
        if o < g // 2:
            continue
        # Check that adding this generator doesn't create short relations
        # (heuristic: just accept if order is large enough)
        gens.append(e)
    if len(gens) < half:
        return None
    # Build adjacency
    adj = [set() for _ in range(n)]
    for i, e in enumerate(elems):
        for g_elem in gens:
            f1 = mat_mult(e, g_elem)
            f2 = mat_mult(e, mat_inv(g_elem))
            j1 = idx[f1]
            j2 = idx[f2]
            if j1 != i:
                adj[i].add(j1)
            if j2 != i:
                adj[i].add(j2)
    # Verify regularity
    for i in range(n):
        if len(adj[i]) != k:
            return None
    return adj

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.time() + 145.0

    # Baseline: LUW
    try:
        n_luw, edges_luw = _luw_construct(k, g)
    except Exception:
        n_luw, edges_luw = None, None

    best_n = n_luw if n_luw is not None else float('inf')
    best_edges = edges_luw if edges_luw is not None else []

    if best_n is None or best_n == float('inf'):
        return 0, []

    rng = random.Random(42)

    # Strategy 1: Field Cayley graphs (k must be odd)
    if k % 2 == 1:
        # Try several primes
        for p in range(max(2*k+3, 50), min(2000, 3*best_n)):
            if _deadline_reached():
                break
            if not _is_prime(p):
                continue
            adj = _cayley_field(k, g, p, rng)
            if adj is None:
                continue
            n = p
            if n < best_n:
                # Quick local girth check from a few vertices
                ok = True
                for s in range(0, min(n, 5), max(1, n//5)):
                    if not _local_girth_ok(adj, n, k, g, s):
                        ok = False
                        break
                if ok:
                    # Full certification
                    if _girth_ok(adj, n, k, g):
                        best_n = n
                        best_edges = _adj_to_edges(adj, n)
                        return best_n, best_edges

    # Strategy 2: SL(2,p) Cayley graphs (k must be odd, p small)
    if k % 2 == 1:
        for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if _deadline_reached():
                break
            n_sl = p * (p*p - 1)
            if n_sl >= best_n:
                continue
            adj = _cayley_sl2(k, g, p, rng)
            if adj is None:
                continue
            if n_sl < best_n:
                ok = True
                for s in range(0, min(n_sl, 5), max(1, n_sl//5)):
                    if not _local_girth_ok(adj, n_sl, k, g, s):
                        ok = False
                        break
                if ok:
                    if _girth_ok(adj, n_sl, k, g):
                        best_n = n_sl
                        best_edges = _adj_to_edges(adj, n_sl)
                        return best_n, best_edges

    # Fallback: randomized search (keep from seed for even k or when algebraic fails)
    if not _deadline_reached():
        # Moore bound
        if g >= 4:
            if g % 2 == 0:
                lb = 2 * (k - 1) * ((k - 1) ** (g // 2) - 1) // (k - 2) if k > 2 else 2 * (k - 1) ** (g // 2)
            else:
                lb = (k - 1) * ((k - 1) ** ((g + 1) // 2) - 1) // (k - 2) + 1 if k > 2 else (k - 1) ** ((g + 1) // 2)
            lb = max(lb, 2 * k)
        else:
            lb = 2 * k

        step = 2 if k % 2 == 1 else 1
        lo = max(lb, k + 1)
        if lo % step:
            lo += step
        hi = min(best_n - 1, lo + 150)

        if hi >= lo:
            n = lo
            while n <= hi and not _deadline_reached():
                if n < best_n:
                    adj = _try_random(n, k, rng)
                    if adj is not None:
                        _refine_fast(adj, n, k, g, rng, steps=200)
                        if _girth_ok(adj, n, k, g):
                            best_n = n
                            best_edges = _adj_to_edges(adj, n)
                            return best_n, best_edges
                n += step

    return best_n, best_edges

def _try_random(n, k, rng):
    if n * k % 2 != 0 or n < k + 1:
        return None
    for _ in range(50):
        stubs = []
        for v in range(n):
            stubs.extend([v] * k)
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
            return adj
    return None

def _refine_fast(adj, n, k, g, rng, steps):
    for _ in range(steps):
        i = rng.randrange(n)
        j_list = list(adj[i])
        if len(j_list) < 2:
            continue
        j = rng.choice(j_list)
        a = rng.randrange(n)
        if a == i or a == j:
            continue
        b_list = list(adj[a])
        if not b_list:
            continue
        b = rng.choice(b_list)
        if b == i or b == j or b == a or b in adj[i] or j in adj[a] or a in adj[i] or i in adj[a]:
            continue
        adj[i].remove(j)
        adj[j].remove(i)
        adj[a].remove(b)
        adj[b].remove(a)
        adj[i].add(b)
        adj[b].add(i)
        adj[a].add(j)
        adj[j].add(a)
        ok = True
        for s in (i, j, a, b):
            if not _local_girth_ok(adj, n, k, g, s):
                ok = False
                break
        if not ok:
            adj[i].remove(b)
            adj[b].remove(i)
            adj[a].remove(j)
            adj[j].remove(a)
            adj[i].add(j)
            adj[j].add(i)
            adj[a].add(b)
            adj[b].add(a)
    return adj
