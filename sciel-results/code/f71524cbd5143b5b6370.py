import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

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
        """Modular inverse using extended Euclidean algorithm."""
        if m <= 1:
            return None
        a = a % m
        if a == 0:
            return None
        try:
            return pow(a, -1, m)
        except ValueError:
            pass
        old_r, r = a, m
        old_s, s = 1, 0
        while r != 0:
            q = old_r // r
            old_r, r = r, old_r - q * r
            old_s, s = s, old_s - q * s
        if old_r != 1:
            return None
        return old_s % m

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:  # g = 2r + 1
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:  # g = 2r
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    # ---------- graph validation ----------
    def check_girth_fast(adj, n, g):
        """
        Check girth using BFS from each vertex with early termination.
        Uses non-backtracking walk certification for exact girth check.
        """
        half_g = g // 2
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                if dist[x] >= half_g:
                    continue
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        if dist[y] < half_g:
                            queue.append(y)
                    else:
                        if y != parent[x] and parent[y] != x:
                            cycle_len = dist[x] + dist[y] + 1
                            if cycle_len < g:
                                return False
        return True

    def is_connected(adj, n):
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
        if not check_girth_fast(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline (deterministic) ----------
    def luw_construct(k, g):
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

    # ---------- PGL(2, q) machinery ----------
    def action_on_point(x, a, b, c, d, q):
        if x == q:  # infinity
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
        """Check if Möbius transform has no fixed points on P^1(F_q)."""
        a, b, c, d = a % q, b % q, c % q, d % q
        if c == 0:
            return False
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False
        # Check if disc is a quadratic residue mod q
        if is_prime(q):
            if pow(disc, (q - 1) // 2, q) == 1:
                return False
        else:
            # For prime powers, use a conservative check
            # Try to find sqrt of disc
            for t in range(q):
                if (t * t) % q == disc:
                    return False
        return True

    def is_involution_matrix(a, b, c, d, q):
        """Check if matrix is an involution (M^2 = I)."""
        a2 = (a*a + b*c) % q
        b2 = (b*(a+d)) % q
        c2 = (c*(a+d)) % q
        d2 = (d*d + b*c) % q
        return a2 == 1 and b2 == 0 and c2 == 0 and d2 == 1

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
        return validate(adj, n, k, g)

    # ---------- Systematic involution search in PGL(2, q) ----------
    def systematic_involution_search(q, k, g, budget):
        """
        Systematically search for sets of k involutions in PGL(2, q)
        that generate a graph with girth >= g.
        """
        n = q + 1
        if n < n_min or n >= k * 2:  # quick sanity check
            return None

        # Generate candidate involutions
        # An involution in PGL(2,q) has the form:
        # For odd q: trace = 0, so a + d = 0, i.e., d = -a
        # Matrix: [[a, b], [c, -a]] with a^2 + bc = 1 (det = -a^2 - bc = -1, so up to scalar)
        # Actually for PGL, we work with projective classes.
        
        candidates = []
        seen = set()
        
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    d = (-a) % q
                    # Check if it's an involution: M^2 should be scalar
                    # M = [[a, b], [c, -a]]
                    # M^2 = [[a^2+bc, 0], [0, a^2+bc]]
                    # For this to be identity (up to scalar), we need a^2 + bc = 1 (mod q)
                    if (a*a + b*c) % q != 1:
                        continue
                    # Check derangement
                    if not is_derangement(a, b, c, d, q):
                        continue
                    # Normalize: make first non-zero entry 1
                    key = (a, b, c, d)
                    # Try to normalize
                    if a != 0:
                        inv_a = modinv(a, q)
                        if inv_a is not None:
                            na = (a * inv_a) % q
                            nb = (b * inv_a) % q
                            nc = (c * inv_a) % q
                            nd = (d * inv_a) % q
                            key = (na, nb, nc, nd)
                    elif b != 0:
                        inv_b = modinv(b, q)
                        if inv_b is not None:
                            na = (a * inv_b) % q
                            nb = (b * inv_b) % q
                            nc = (c * inv_b) % q
                            nd = (d * inv_b) % q
                            key = (na, nb, nc, nd)
                    elif c != 0:
                        inv_c = modinv(c, q)
                        if inv_c is not None:
                            na = (a * inv_c) % q
                            nb = (b * inv_c) % q
                            nc = (c * inv_c) % q
                            nd = (d * inv_c) % q
                            key = (na, nb, nc, nd)
                    
                    if key not in seen:
                        seen.add(key)
                        candidates.append((a, b, c, d))
        
        if len(candidates) < k:
            return None
        
        # Random sampling of k-tuples from candidates
        rng = random.Random()
        attempts = 0
        max_attempts = 50000 if q < 50 else 20000
        
        while time.time() < budget and attempts < max_attempts:
            attempts += 1
            # Sample k distinct candidates
            idxs = rng.sample(range(len(candidates)), k)
            gens = [candidates[i] for i in idxs]
            
            # Quick check: all must be derangements (already ensured)
            res = build_from_gens(gens, q, k, g)
            if res:
                return res
        
        return None

    # ---------- Random search in PGL(2, q) ----------
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
            """Generate k generators, preferring involutions for better girth."""
            out = []
            attempts = 0
            while len(out) < k and attempts < k * 100:
                attempts += 1
                m = random_gen()
                mi = inv_matrix(*m, q)
                if mi is None:
                    continue
                if not is_derangement(*m, q):
                    continue
                # Check if involution
                is_inv = is_involution_matrix(*m, q)
                
                if is_inv or random.random() < 0.3:
                    out.append(m)
                    if mi != m and len(out) < k:
                        out.append(mi)
            while len(out) < k:
                m = random_gen()
                if is_derangement(*m, q):
                    out.append(m)
            return out[:k]

        seen = set()
        while time.time() < budget:
            gens = random_gens()
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            if not all(is_derangement(*gen, q) for gen in gens):
                continue
            res = build_from_gens(gens, q, k, g)
            if res:
                return res
        return None

    # ---------- Voltage lift approach ----------
    def voltage_lift_search(base_n, k, g, budget):
        """
        Try to construct a k-regular graph with girth >= g by lifting
        a smaller base graph with voltages.
        """
        # This is a fallback; the PGL approach is preferred
        # For now, skip this as it's complex and may not be better
        return None

    # ---------- main ----------
    # 1. deterministic baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # 2. Systematic involution search in PGL(2, q) for small q
    # Try q values from Moore bound upward
    q_start = max(3, n_min - 1)
    q_end = min(best_n[0] - 1, 200)  # Limit q range for practicality
    
    # First pass: systematic involution search for small q
    for q in range(q_start, min(q_start + 30, q_end + 1)):
        if time.time() > deadline - 30.0:
            break
        if q % 2 == 0:
            continue
        if not is_prime_power(q):
            continue
        n = q + 1
        if n < n_min or n >= best_n[0]:
            continue
        
        # Allocate time based on q
        time_left = deadline - 30.0 - time.time()
        if time_left <= 0:
            break
        
        # Smaller q get more time
        budget = min(5.0, time_left * 0.5)
        
        res = systematic_involution_search(q, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            # Found improvement; continue to see if we can do better
            # But smaller q already tried, so we can stop
            # Actually, we're going upward in q, so this is the smallest q that worked
            # We could try to find even smaller, but we started from n_min
            # So this should be optimal for this approach
            # Let's do a quick random search to confirm
            break

    # 3. Random search in PGL(2, q) for remaining time
    if time.time() < deadline - 10.0:
        # Try a few more q values with random search
        for q in range(q_start, min(q_start + 50, q_end + 1)):
            if time.time() > deadline - 10.0:
                break
            if q % 2 == 0:
                continue
            if not is_prime_power(q):
                continue
            n = q + 1
            if n < n_min or n >= best_n[0]:
                continue
            
            time_left = deadline - 10.0 - time.time()
            if time_left <= 0:
                break
            
            budget = min(2.0, time_left * 0.3)
            
            res = random_search_pgl(q, k, g, budget)
            if res and res[0] < best_n[0]:
                best_n[0] = res[0]
                best_result = res
                break

    return best_result

# EVOLVE-BLOCK-END
