import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin vs 120s limit

    # Moore bound
    def moore_bound(k, g):
        if g % 2 == 0:
            h = g // 2 - 1
            return k * ((k - 1) ** h - 1) // (k - 2) + 1
        else:
            h = (g - 1) // 2
            return 1 + k * ((k - 1) ** h - 1) // (k - 2)

    mb = moore_bound(k, g)

    # ---------- LUW construction (guaranteed fallback) ----------
    def luw_construct(k, g):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def gf_add(x, y): return x ^ y
            def gf_sub(x, y): return x ^ y
            def gf_mul(x, y): return table[x][y]
        else:
            def gf_add(x, y): return (x + y) % q
            def gf_sub(x, y): return (x - y) % q
            def gf_mul(x, y): return x * y % q

        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

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
                        y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                    else:
                        y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        queue = [0]
        head = 0
        while head < len(order):
            if time.time() > deadline - 2:
                break
            v_idx = queue[head]
            head += 1
            v = order[v_idx]
            for other in neighbours(v):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                    queue.append(len(order) - 1)
                if v[0] == 0:
                    edges.append((v_idx, index[other]))
        return len(order), edges

    luw_n, luw_edges = luw_construct(k, g)

    # ---------- Girth verification (non-backtracking walks from every vertex) ----------
    def check_girth(n, adj, g):
        for s in range(n):
            dist = {}
            q2 = deque()
            for v in adj[s]:
                q2.append((v, s))
                dist[(v, s)] = 1
            while q2:
                cur, prev = q2.popleft()
                d_cur = dist[(cur, prev)]
                if d_cur >= g - 1:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    if nxt == s:
                        if d_cur + 1 < g:
                            return False
                    else:
                        if (nxt, cur) not in dist:
                            dist[(nxt, cur)] = d_cur + 1
                            q2.append((nxt, cur))
        return True

    # Verify LUW (it should be valid, but be safe)
    adj_luw = [[] for _ in range(luw_n)]
    for u, v in luw_edges:
        adj_luw[u].append(v)
        adj_luw[v].append(u)
    luw_ok = check_girth(luw_n, adj_luw, g)

    if not luw_ok:
        return luw_n, luw_edges

    best_n = luw_n
    best_edges = luw_edges

    # ---------- Fast circulant search using non-backtracking coefficient set ----------
    # A 4-regular circulant graph C(n, {a, b, n-a, n-b}) has girth >= g iff
    # no non-backtracking walk of length < g sums to 0 mod n.
    # Precompute the set of (c1, c2) coefficients for non-backtracking walks of length 3..g-1.
    valid_coeffs = set()
    if g > 3:
        states = set()
        states.add((1, 1, 0))
        states.add((-1, -1, 0))
        states.add((2, 0, 1))
        states.add((-2, 0, -1))
        for L in range(2, g):
            new_states = set()
            for last_step, c1, c2 in states:
                for step in (1, -1, 2, -2):
                    if step == -last_step:
                        continue
                    nc1 = c1 + (1 if step == 1 else -1 if step == -1 else 0)
                    nc2 = c2 + (1 if step == 2 else -1 if step == -2 else 0)
                    new_states.add((step, nc1, nc2))
            states = new_states
            if L >= 3:
                for _, c1, c2 in states:
                    valid_coeffs.add((c1, c2))

    def circulant_girth_ok(n, a, b):
        for c1, c2 in valid_coeffs:
            if (c1 * a + c2 * b) % n == 0:
                return False
        return True

    def build_circulant_edges(n, a, b):
        edges = set()
        for i in range(n):
            j1 = (i + a) % n
            edges.add((i, j1) if i < j1 else (j1, i))
            j2 = (i + b) % n
            edges.add((i, j2) if i < j2 else (j2, i))
        return list(edges)

    # Search n from Moore bound upward for a valid circulant
    if best_n > mb:
        n_max_search = min(best_n - 1, mb + 300)
        found_circ = False
        for n in range(mb, n_max_search + 1):
            if time.time() > deadline - 5:
                break
            # Deterministic sweep of small offsets first
            half = n // 2 - 1
            max_off = min(200, half)
            for a in range(1, max_off + 1):
                if time.time() > deadline - 5:
                    break
                for b in range(a + 1, max_off + 1):
                    if circulant_girth_ok(n, a, b):
                        best_n = n
                        best_edges = build_circulant_edges(n, a, b)
                        found_circ = True
                        break
                if found_circ:
                    break
            if found_circ:
                break
            # Random offsets for this n
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                a = random.randint(1, half)
                b = random.randint(1, half)
                if a == b:
                    continue
                if circulant_girth_ok(n, a, b):
                    best_n = n
                    best_edges = build_circulant_edges(n, a, b)
                    found_circ = True
                    break
            if found_circ:
                break

    # ---------- PSL(2,p) Cayley graph search ----------
    # For prime p, PSL(2,p) has order p(p^2-1)/gcd(2,p-1).
    # For odd p, order = p(p^2-1)/2. We look for primes where this is near Moore bound.
    # Use 4 involutory generators (elements of order 2) to form a 4-regular Cayley graph.
    # In PSL(2,p) for odd p, involutions are conjugate to [[0,-1],[1,0]] (order 4 in PGL, order 2 in PSL).
    # Actually in PSL(2,p), elements of order 2 have trace 0.
    # A 4-regular Cayley graph needs a set S of 4 elements with s=s^-1 (involutions) and S closed under inverse.
    # For PSL(2,p) with p ≡ 1 (mod 4), we can use specific involutions.
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def mat_mul(A, B):
        a11, a12, a21, a22 = A
        b11, b12, b21, b22 = B
        return ((a11*b11 + a12*b21) % p,
                (a11*b12 + a12*b22) % p,
                (a21*b11 + a22*b21) % p,
                (a21*b12 + a22*b22) % p)
    
    def mat_trace(A):
        return (A[0] + A[3]) % p
    
    def mat_eq(A, B):
        return A[0] == B[0] and A[1] == B[1] and A[2] == B[2] and A[3] == B[3]
    
    def psl_order(p):
        if p % 2 == 1:
            return p * (p*p - 1) // 2
        return p * (p*p - 1)
    
    if best_n > mb and time.time() < deadline - 15:
        # Find primes p where psl_order(p) is in range [mb, best_n-1]
        p_start = 3
        # Estimate p from mb: p^3/2 ≈ mb => p ≈ (2*mb)^(1/3)
        p_est = max(3, int((2 * mb) ** (1/3)) - 2)
        p_end = min(200, int((2 * (best_n - 1)) ** (1/3)) + 5)
        
        found_psl = False
        for p in range(max(3, p_est), p_end + 1):
            if time.time() > deadline - 15:
                break
            if not is_prime(p):
                continue
            group_n = psl_order(p)
            if group_n < mb or group_n >= best_n:
                continue
            
            # In PSL(2,p) for odd p, involutions are matrices with trace 0 and determinant 1.
            # [[a, b], [c, -a]] with a^2 + bc = 1.
            # Standard involution: [[0, -1], [1, 0]] which has order 2 in PSL(2,p) when p is odd.
            # Actually [[0,-1],[1,0]]^2 = [[-1,0],[0,-1]] = identity in PSL(2,p) since we mod out by ±I.
            
            # We need 4 distinct involutions that generate a high-girth graph.
            # Let's use: e1 = [[0,-1],[1,0]], and conjugates by powers of a rotation.
            # Rotation: R = [[0,1],[-1,0]] ... wait that's the same.
            # Let's use R = [[1,1],[0,1]] (unipotent, order p).
            # Conjugate: R^k * e1 * R^(-k)
            
            # For simplicity, try a few fixed generator sets and check girth.
            # e1 = (0, -1, 1, 0)  [i.e., [[0,-1],[1,0]]]
            # e2 = (1, 0, 0, -1) ... wait, trace must be 0: (1,0,0,-1) has trace 0, det = -1. Not in SL(2).
            # Need det = 1: a^2 + bc = 1.
            # e1 = (0, -1, 1, 0): 0 + (-1)(1) = -1 ≠ 1. Oops.
            # In SL(2,p): [[a,b],[c,d]] with ad-bc=1.
            # [[0,-1],[1,0]]: 0*0 - (-1)*1 = 1. Yes! Trace = 0. Good.
            # e2 = [[1,0],[0,1]] is identity, not useful.
            # Another: [[1,1],[1,-1]]: 1*(-1) - 1*1 = -2. Not 1 unless p=2.
            # [[a,b],[c,-a]] with a^2+bc=1.
            # e1 = (0, -1, 1, 0)
            # e2 = (1, 0, 0, -1)? det = -1. No.
            # Let's use e1 = (0, -1, 1, 0), and for others use conjugates.
            
            # Simpler approach: just try a few candidate 4-element involution sets.
            # For PSL(2,p), all involutions are conjugate. Pick e = (0, -1, 1, 0).
            # Generate conjugates by unipotent matrices U_k = (1, k, 0, 1).
            # U_k * e * U_k^{-1} = U_k * e * (1, -k, 0, 1)
            
            # Compute U_k * e:
            # (1,k,0,1) * (0,-1,1,0) = (1*0+k*1, 1*(-1)+k*0, 0*0+1*1, 0*(-1)+1*0) = (k, -1, 1, 0)
            # Then * (1,-k,0,1):
            # (k,-1,1,0) * (1,-k,0,1) = (k*1+(-1)*0, k*(-k)+(-1)*1, 1*1+0*0, 1*(-k)+0*1) = (k, -k^2-1, 1, -k)
            # Check det: k*(-k) - (-k^2-1)*1 = -k^2 + k^2 + 1 = 1. Good.
            # Trace: k + (-k) = 0. Good, it's an involution.
            
            # So conjugate of e by U_k is (k, -k^2-1, 1, -k) mod p.
            # For k=0: (0, -1, 1, 0) = e.
            # For k=1: (1, -2, 1, -1).
            # For k=2: (2, -5, 1, -2).
            # These are distinct for different k (as long as p > 4 or so).
            
            # Try sets of 4: {e_{k1}, e_{k2}, e_{k3}, e_{k4}} for various choices.
            # The Cayley graph vertex is a group element (matrix). Edge from g to g*s for s in S.
            # We need to enumerate the group and check girth.
            # Group size can be up to ~best_n which could be ~1000. Enumeration via BFS from identity.
            
            # Actually, let's just try a few specific small sets and verify.
            # For efficiency, limit to group_n <= 600 (since girth check is O(n * (k-1)^(g-1)) worst case).
            if group_n > 600:
                continue
            
            # Generate group elements via BFS from identity using the 4 generators
            # First, define the 4 generators
            gens = []
            for kk in [0, 1, 2, 3]:
                a = kk % p
                b = (-a*a - 1) % p
                c = 1 % p
                d = (-a) % p
                gens.append((a, b, c, d))
            
            # Check they are distinct
            if len(set(gens)) < 4:
                continue
            
            # BFS to enumerate group
            identity = (1, 0, 0, 1)
            elem_to_idx = {identity: 0}
            order_list = [identity]
            bfs_q = [0]
            head_idx = 0
            psl_edges = []
            
            while head_idx < len(order_list):
                if time.time() > deadline - 15:
                    break
                vi = bfs_q[head_idx]
                head_idx += 1
                elem = order_list[vi]
                for gi, gen in enumerate(gens):
                    prod = mat_mul(elem, gen)
                    # Also check if prod is in PSL (det should be 1, which it is)
                    # In PSL(2,p), elements are equivalence classes mod ±I
                    # So prod and -prod are the same element
                    neg_prod = ((-prod[0]) % p, (-prod[1]) % p, (-prod[2]) % p, (-prod[3]) % p)
                    key = prod if prod < neg_prod else neg_prod
                    if key not in elem_to_idx:
                        elem_to_idx[key] = len(order_list)
                        order_list.append(key)
                        bfs_q.append(len(order_list) - 1)
                    # Add edge (only add once per undirected edge)
                    neighbor_idx = elem_to_idx[key]
                    if vi < neighbor_idx:
                        psl_edges.append((vi, neighbor_idx))
            
            # Check if we got the full group
            if len(order_list) != group_n:
                continue
            
            # Verify it's 4-regular and check girth
            adj_psl = [[] for _ in range(group_n)]
            for u, v in psl_edges:
                adj_psl[u].append(v)
                adj_psl[v].append(u)
            
            if any(len(adj_psl[v]) != k for v in range(group_n)):
                continue
            
            if check_girth(group_n, adj_psl, g):
                best_n = group_n
                best_edges = psl_edges
                found_psl = True
                break
        
        if found_psl:
            # We found a better graph, skip remaining searches
            return best_n, best_edges

    # ---------- Secondary: random configuration-model search for non-circulant graphs ----------
    if best_n > mb and time.time() < deadline - 10:
        n_max_rand = min(best_n - 1, mb + 100)
        for n in range(mb, n_max_rand + 1):
            if time.time() > deadline - 10:
                break
            found_rand = False
            for trial in range(30):
                if time.time() > deadline - 10:
                    break
                random.seed(trial * n + 12345)
                stubs = []
                for i in range(n):
                    for _ in range(k):
                        stubs.append(i)
                random.shuffle(stubs)
                edges_set = set()
                valid = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v:
                        valid = False
                        break
                    key = (u, v) if u < v else (v, u)
                    if key in edges_set:
                        valid = False
                        break
                    edges_set.add(key)
                if not valid:
                    continue
                adj2 = [[] for _ in range(n)]
                for u, v in edges_set:
                    adj2[u].append(v)
                    adj2[v].append(u)
                if any(len(adj2[v]) != k for v in range(n)):
                    continue
                if check_girth(n, adj2, g):
                    best_n = n
                    best_edges = list(edges_set)
                    found_rand = True
                    break
            if found_rand:
                break

    return best_n, best_edges
# EVOLVE-BLOCK-END
