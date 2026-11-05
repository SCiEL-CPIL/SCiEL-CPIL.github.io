import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

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

    def is_quadratic_residue(a, q):
        """Check if a is a quadratic residue mod q (works for prime powers)."""
        a = a % q
        if a == 0:
            return True
        if not is_prime_power(q):
            return True  # conservative: assume residue
        # Find prime base p of q
        p = None
        for cand in range(2, int(math.isqrt(q)) + 1):
            if q % cand == 0:
                t = q
                while t % cand == 0:
                    t //= cand
                if t == 1:
                    p = cand
                    break
        if p is None:
            p = q  # q is prime
        # Reduce to mod p
        a_p = a % p
        if a_p == 0:
            return True
        # Euler's criterion mod p (p is prime)
        return pow(a_p, (p - 1) // 2, p) == 1

    def is_nonsquare(a, q):
        """Check if a is a non-residue mod q (q must be odd prime)."""
        a = a % q
        if a == 0:
            return False
        return pow(a, (q - 1) // 2, q) == q - 1

    def modinv(a, m):
        """Extended Euclidean algorithm for modular inverse."""
        if m <= 1:
            return None
        a = a % m
        if a == 0:
            return None
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
        """Check girth using BFS from each vertex."""
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        if dist[y] < g:
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

    # ---------- Lazebnik-Ustimenko-Woldar baseline ----------
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
        if is_quadratic_residue(disc, q):
            return False
        return True

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
        return validate(adj, n, k, g)

    # ---------- main ----------
    # 1. deterministic baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # 2. Structured PGL(2, q) search
    def try_structured_pgl(q, k, g, budget):
        """Try to find a k-regular graph of girth >= g using PGL(2,q) with structured generators."""
        n = q + 1
        if n < n_min or n >= best_n[0]:
            return None
        if q % 2 == 0 or not is_prime_power(q):
            return None
        
        # For odd prime q, we can use a more structured approach
        # Look for involutions (order 2 elements) that are derangements
        # An element (a,b,c,d) is an involution iff (a+d)^2 - 4(ad-bc) is a nonsquare (or 0 with care)
        
        def find_involutions(q):
            """Find involutions in PGL(2,q) that are derangements."""
            invs = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        for d in range(q):
                            det = (a * d - b * c) % q
                            if det == 0:
                                continue
                            # Check if it's an involution: m^2 = I
                            # m^2 = (a^2+bc, b(a+d), c(a+d), d^2+bc)
                            a2 = (a * a + b * c) % q
                            b2 = (b * (a + d)) % q
                            c2 = (c * (a + d)) % q
                            d2 = (d * d + b * c) % q
                            if a2 == 1 and b2 == 0 and c2 == 0 and d2 == 1:
                                # It's an involution. Check if derangement.
                                if is_derangement(a, b, c, d, q):
                                    invs.append((a, b, c, d))
            return invs
        
        # Find all involutions that are derangements
        invs = find_involutions(q)
        
        if len(invs) < k:
            # Not enough involutions, try random search
            return random_search_pgl(q, k, g, budget)
        
        # Try combinations of k involutions
        # Use a greedy approach: start with the first involution, then add others
        # that don't create short cycles
        
        def check_short_cycle(adj, n, max_len):
            """Check if there's a cycle shorter than max_len using BFS."""
            for u in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[u] = 0
                queue = deque([u])
                while queue:
                    x = queue.popleft()
                    for y in adj[x]:
                        if dist[y] == -1:
                            dist[y] = dist[x] + 1
                            parent[y] = x
                            if dist[y] < max_len:
                                queue.append(y)
                        else:
                            if y != parent[x] and parent[y] != x:
                                cycle_len = dist[x] + dist[y] + 1
                                if cycle_len < max_len:
                                    return True
            return False
        
        # Try random subsets of involutions
        seen = set()
        while time.time() < budget and len(seen) < 500:
            # Randomly pick k involutions
            idxs = random.sample(range(len(invs)), k)
            gens = [invs[i] for i in idxs]
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            
            # Build graph
            n = q + 1
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(n):
                x = i if i < q else q
                for gen in gens:
                    y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                    if y is None or y == i:
                        valid = False
                        break
                    j = y if y < q else q
                    if j in adj[i]:
                        # Duplicate edge - not allowed for simple graph
                        valid = False
                        break
                    adj[i].add(j)
                    adj[j].add(i)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check degree
            if any(len(adj[i]) != k for i in range(n)):
                continue
            
            # Check connectivity
            if not is_connected(adj, n):
                continue
            
            # Check girth
            if not check_girth_fast(adj, n, g):
                continue
            
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return (n, edges)
        
        return None
    
    def random_search_pgl(q, k, g, budget):
        """Random search in PGL(2, q) for generators."""
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
                # Prefer involutions
                a2 = (m[0]*m[0] + m[1]*m[2]) % q
                b2 = (m[1]*(m[0]+m[3])) % q
                c2 = (m[2]*(m[0]+m[3])) % q
                d2 = (m[3]*m[3] + m[1]*m[2]) % q
                is_involution = (a2 == 1 and b2 == 0 and c2 == 0 and d2 == 1)
                
                if is_involution or random.random() < 0.3:
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

    # Search q values from Moore bound up to best_n - 1
    q_cursor = max(3, n_min - 1)
    max_q = best_n[0] - 1
    
    while q_cursor < max_q and time.time() < deadline - 5.0:
        time_left = deadline - 5.0 - time.time()
        if time_left <= 0:
            break
        
        # Give more time to smaller q values
        if q_cursor < n_min + 20:
            budget = min(2.0, time_left / 2)
        elif q_cursor < n_min + 100:
            budget = min(1.0, time_left / 3)
        else:
            budget = min(0.5, time_left / 4)
        
        # Try structured search first (for odd prime q)
        if q_cursor % 2 == 1 and is_prime(q_cursor):
            res = try_structured_pgl(q_cursor, k, g, budget * 0.7)
            if res and res[0] < best_n[0]:
                best_n[0] = res[0]
                best_result = res
                return best_result
        
        # Try random search
        res = random_search_pgl(q_cursor, k, g, budget * 0.3)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            return best_result
        
        q_cursor += 1

    return best_result

# EVOLVE-BLOCK-END
