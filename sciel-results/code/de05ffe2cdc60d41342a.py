import time
import random
import math
from collections import deque, defaultdict

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def remaining():
        return deadline - time.time()

    # ---------- arithmetic helpers ----------
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                return t == 1
        return False

    def modinv(a, m):
        if m <= 1:
            return None
        try:
            return pow(a, -1, m)
        except (ValueError, ZeroDivisionError):
            return None

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    # ---------- graph validation ----------
    def check_girth_nb(adj, n, g):
        for u in range(n):
            dist = {}
            queue = deque()
            for v in adj[u]:
                dist[(v, u)] = 1
                queue.append((v, u))
            while queue:
                x, prev = queue.popleft()
                d = dist[(x, prev)]
                if d >= g - 1:
                    continue
                for y in adj[x]:
                    if y == prev:
                        continue
                    if y == u:
                        if d + 1 < g:
                            return False
                        continue
                    key = (y, x)
                    if key not in dist:
                        dist[key] = d + 1
                        queue.append((y, x))
        return True

    def is_connected(adj, n):
        if n == 0:
            return True
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return False
        if not is_connected(adj, n):
            return False
        if not check_girth_nb(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline ----------
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = sub = lambda x, y: x ^ y
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
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

    # ---------- PGL(2, q) machinery ----------
    def action_on_point(x, a, b, c, d, q):
        if x == q:
            if c == 0:
                return q
            inv_c = modinv(c, q)
            if inv_c is None:
                return None
            return (a * inv_c) % q
        num = (a * x + b) % q
        den = (c * x + d) % q
        if den == 0:
            return q
        inv_den = modinv(den, q)
        if inv_den is None:
            return None
        return (num * inv_den) % q

    def inv_matrix(a, b, c, d, q):
        det = (a * d - b * c) % q
        det_inv = modinv(det, q)
        if det_inv is None:
            return None
        return (d * det_inv % q, -b * det_inv % q, -c * det_inv % q, a * det_inv % q)

    def is_derangement(a, b, c, d, q):
        if c == 0:
            return False
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False
        return pow(disc, (q - 1) // 2, q) != 1

    def build_from_gens(gens, q, k, g):
        n = q + 1
        adj = [set() for _ in range(n)]
        for i in range(n):
            x = i if i < q else q
            for gen in gens:
                y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                if y is None or y == i:
                    return None
                j = y if y < q else q
                adj[i].add(j)
                adj[j].add(i)
        for i in range(n):
            if len(adj[i]) != k:
                return None
        adj_list = [sorted(adj[i]) for i in range(n)]
        return validate(adj_list, n, k, g)

    def try_structured_pgl(q, k, g):
        if q % 2 == 0 or not is_prime_power(q):
            return None
        n = q + 1
        if n < n_min:
            return None

        def is_nonsquare(a):
            a %= q
            if a == 0:
                return True
            return pow(a, (q - 1) // 2, q) != 1

        cands = []
        for cc in range(1, q):
            if is_nonsquare(1 + 4 * cc):
                cands.append(cc)
            if len(cands) >= 6:
                break
        if len(cands) < 2:
            return None

        h_ok = is_nonsquare(8)
        h2_cands = []
        for cc in range(1, q):
            if is_nonsquare(1 - cc * cc):
                h2_cands.append(cc)
            if len(h2_cands) >= 4:
                break

        c1, c2 = cands[0], cands[1]
        g1 = (1, c1, 1, 0)
        g1i = inv_matrix(*g1, q)
        g2 = (1, c2, 1, 0)
        g2i = inv_matrix(*g2, q)
        if g1i is None or g2i is None:
            return None
        h = (1, 1, 1, q - 1)

        if h_ok:
            res = build_from_gens([g1, g1i, g2, g2i, h], q, k, g)
            if res:
                return res

        if len(h2_cands) >= 1:
            cc = h2_cands[0]
            g3 = (1, q - cc, 1, cc)
            res = build_from_gens([g1, g1i, g2, g2i, g3], q, k, g)
            if res:
                return res
        return None

    # ---------- Random PGL search ----------
    def random_search_pgl(q, k, g, budget):
        n = q + 1
        if n < n_min or n >= best_n[0]:
            return None
        if q % 2 == 0 or not is_prime_power(q):
            return None

        def random_gen():
            while True:
                a = random.randrange(q)
                b = random.randrange(q)
                c = random.randrange(q)
                d = random.randrange(q)
                if (a * d - b * c) % q != 0:
                    return (a, b, c, d)

        def random_gens():
            out = []
            while len(out) < k:
                m = random_gen()
                if not is_derangement(*m, q):
                    continue
                if random.random() < 0.5:
                    mi = inv_matrix(*m, q)
                    if mi is not None and is_derangement(*mi, q):
                        out.extend([m, mi])
                        if len(out) > k:
                            out = out[:k]
                else:
                    out.append(m)
            return out

        seen = set()
        while time.time() < budget:
            gens = random_gens()
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            res = build_from_gens(gens, q, k, g)
            if res:
                return res
        return None

    # ---------- Benson construction via unitriangular groups ----------
    def gf_mul_table(q):
        """Build multiplication table for GF(q) where q is a prime power."""
        if is_prime(q):
            return [[(a * b) % q for b in range(q)] for a in range(q)]
        # For prime powers, use a primitive element to build the field
        # This is a simplified approach; for this task q is typically prime
        return None

    def benson_construct(k, g):
        """
        Benson construction: use a group of unitriangular matrices over GF(q)
        to construct a k-regular graph with girth g.
        
        For a generalized quadrangle GQ(s,t), the Benson construction uses
        a group G of order (s+1)^3 acting on s^2 + s + 1 points.
        
        More generally, for odd girth g = 2r+1, we look for q such that
        q ≈ k, and construct a graph with n = q^2 + q + 1 vertices.
        """
        # For girth g, we need q such that the construction works
        # The Benson construction for GQ gives girth 4 (bipartite)
        # For odd girth, we use the generalized polygon construction
        
        # Try q values around k
        # n = q^2 + q + 1 for the standard construction
        # But we need n >= moore_bound and the graph must be k-regular
        
        best = None
        for q in range(max(2, k - 2), k + 5):
            if not is_prime_power(q):
                continue
            n = q * q + q + 1
            if n < n_min:
                continue
            if best is not None and n >= best[0]:
                continue
            
            # Build the Benson graph
            # Vertices are pairs (i, j) with 0 <= i, j < q, plus one extra vertex
            # Actually, the standard Benson construction for GQ(s,t) has:
            # Points: O (origin), (i) for i in GF(q), (i,j) for i,j in GF(q)
            # Total: 1 + q + q^2 = q^2 + q + 1
            # Lines: similar structure
            # The incidence graph is bipartite with girth 4.
            
            # For odd girth, we need a different approach.
            # Let's try the coset graph approach with unitriangular matrices.
            
            # The group U_3(q) consists of upper triangular 3x3 matrices with 1s on diagonal
            # |U_3(q)| = q^3
            # We want a subgroup H of index n such that |U_3(q)| / |H| = n
            # and the double cosets give k neighbors.
            
            # For the standard construction:
            # G = U_3(q), |G| = q^3
            # H = {matrices with top-right entry 0} ∩ ... 
            # Actually, let's use the known Benson construction directly.
            
            # Benson's construction for generalized quadrangles:
            # G = U_3(q), H = Borel subgroup
            # The coset space G/H has q^2 + q + 1 elements
            # Each element has exactly q+1 neighbors in the coset graph
            # This gives a (q+1)-regular graph on q^2+q+1 vertices with girth 4.
            
            # For odd girth, we need a different subgroup or a different group.
            
            # Let's try: for girth g = 2r+1, use q such that k = q+1 or k = q
            # and verify the girth.
            
            if k != q + 1 and k != q:
                continue
            
            # Build the coset graph of U_3(q) / Borel
            # U_3(q) elements: [[1, a, c], [0, 1, b], [0, 0, 1]] for a,b,c in GF(q)
            # Borel subgroup B: elements with c = 0, so [[1, a, 0], [0, 1, b], [0, 0, 1]]
            # |B| = q^2, so |G/B| = q^3 / q^2 = q... that's not right.
            
            # Let me reconsider. The standard Benson construction:
            # G = U_3(q) of order q^3
            # H = {[[1, 0, c], [0, 1, 0], [0, 0, 1]] : c in GF(q)} of order q
            # Then |G/H| = q^2, not q^2+q+1.
            
            # The correct Benson construction uses:
            # G = U_3(q), and H is a subgroup of order q such that
            # the double cosets under a certain set of generators give the right structure.
            
            # Actually, the well-known Benson construction for GQ(s,t) with s=t=q-1:
            # Uses G = U_3(q) and specific subgroups.
            
            # Let me just construct the graph directly using the known formula.
            # Vertices: 0 (infinity), 1..q (points at infinity), (i,j) for 0<=i,j<q
            # Total: 1 + q + q^2 = q^2 + q + 1
            
            # The adjacency for the Benson GQ construction:
            # Vertex 0 is adjacent to vertices 1, 2, ..., q
            # Vertex i (1<=i<=q) is adjacent to 0 and to vertices (j, i*j) for j in GF(q)
            # Vertex (i,j) is adjacent to vertices (i, j') for all j' (q neighbors) 
            #   and to vertices (i', j) for all i' (q neighbors)
            #   Wait, that gives degree 2q, not q+1.
            
            # I'm getting confused. Let me just try a simpler approach:
            # For the specific case where k = q+1 and g = 4 (even), we can use
            # the known construction. For odd girth, we may not get a good result.
            
            # Since the task asks for odd girth g, and Benson constructions typically
            # give even girth (bipartite), this may not be directly applicable.
            # But let's still try and validate.
            
            # Construct the graph with vertices 0..n-1
            # Use the standard GQ Benson construction
            adj = [set() for _ in range(n)]
            
            # Vertex 0 = infinity
            # Vertices 1..q = points at infinity (indexed 1..q)
            # Vertices q+1..q+1+q^2-1 = affine points (i,j) with i,j in 0..q-1
            
            # Map (i,j) to vertex index: q + 1 + i*q + j
            def affine_idx(i, j):
                return q + 1 + i * q + j
            
            # Adjacency:
            # 0 adjacent to 1, 2, ..., q
            for i in range(1, q + 1):
                adj[0].add(i)
                adj[i].add(0)
            
            # i (1<=i<=q) adjacent to (j, i*j mod q) for j in 0..q-1
            for i in range(1, q + 1):
                for j in range(q):
                    jj = (i * j) % q
                    v = affine_idx(j, jj)
                    adj[i].add(v)
                    adj[v].add(i)
            
            # (i,j) adjacent to (i, j') for all j' != j (q-1 neighbors)
            # and to (i', j) for all i' != i (q-1 neighbors)
            # Total degree for affine vertex: q-1 + q-1 = 2q-2
            # Plus the connection to the "infinity" direction... 
            # This doesn't give k = q+1.
            
            # I think I'm misremembering the construction. Let me abort this
            # and rely on the existing methods.
            break
        
        return best

    # ---------- Main ----------
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)
    
    # Try Benson construction
    benson_res = benson_construct(k, g)
    if benson_res and benson_res[0] < best_n[0]:
        best_n[0] = benson_res[0]
        best_result = benson_res

    # Try structured PGL for small q
    q_max_struct = min(best_n[0] - 2, 80)
    for curr_q in range(max(3, n_min - 1), q_max_struct + 1):
        if time.time() > deadline - 30.0:
            break
        res = try_structured_pgl(curr_q, k, g)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            return best_result

    # Random PGL search with improved budget allocation
    q_cursor = max(3, n_min - 1)
    while q_cursor + 1 < best_n[0] and time.time() < deadline - 10.0:
        # Use more time per q value for better search
        budget = min(deadline - 10.0, time.time() + 2.0)
        res = random_search_pgl(q_cursor, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
        q_cursor += 1

    return best_result

# EVOLVE-BLOCK-END
