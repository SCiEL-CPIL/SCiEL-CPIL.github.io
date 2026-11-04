import itertools
import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit
    
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
        
    def is_prime_power(n):
        """Check if n is a prime power p^e for e >= 1."""
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                if t == 1:
                    return True
                return False
        return False

    def modinv(a, m):
        """Compute modular inverse of a mod m. Returns None if not invertible."""
        if m <= 1:
            return None
        try:
            return pow(a, -1, m)
        except ValueError:
            return None
            
    def is_quadratic_residue(a, q):
        """Check if a is a quadratic residue mod q (for odd prime powers)."""
        if q % 2 == 0:
            # For q=2^e, every element is a square? No.
            # But we are likely dealing with odd q for PGL(2, q) derangements of this type.
            # For now, assume q is odd.
            pass
        if a % q == 0:
            return True # 0 is a square
        return pow(a, (q - 1) // 2, q) == 1

    def moore_bound(k, g):
        if k <= 1: return 0
        if g == 1: return 1
        r = g // 2
        if g % 2 == 1:
            # g = 2r + 1
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            # g = 2r
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)
    
    def action_on_point(x, a, b, c, d, q):
        if x == q:  # infinity
            if c == 0:
                return q
            else:
                inv_c = modinv(c, q)
                if inv_c is None:
                    return None
                return (a * inv_c) % q
        else:
            num = (a * x + b) % q
            den = (c * x + d) % q
            if den == 0:
                return q
            else:
                inv_den = modinv(den, q)
                if inv_den is None:
                    return None
                return (num * inv_den) % q

    def check_girth(adj, n, g):
        """
        Returns True if girth >= g.
        Uses BFS from each vertex.
        """
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
                        queue.append(y)
                    elif y != parent[x] and parent[y] != x:
                        # Found a cycle
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < g:
                            return False
        return True

    def try_structured_pgl(q, k, g):
        """
        Try to build a k-regular graph of girth >= g on q+1 vertices
        using structured derangements in PGL(2, q).
        """
        if q % 2 == 0:
            return None # Skip even q for now, focus on odd prime powers
        
        n = q + 1
        
        # We need k generators.
        # Strategy: Use pairs of inverse derangements + 1 involution/derangement.
        # G_k: x -> (x+k)/x. Derangement if 1+4k is non-square.
        # H: x -> (x+1)/(x-1). Derangement if 8 is non-square.
        
        candidates = []
        
        # Find k values such that 1+4k is a non-square
        for kk in range(1, q):
            disc = (1 + 4 * kk) % q
            if not is_quadratic_residue(disc, q):
                candidates.append(kk)
                if len(candidates) >= 10:
                    break
        
        if len(candidates) < 2:
            return None
            
        # We need to pick generators.
        # Let's try combinations of 2 pairs (4 gens) + 1 more.
        # Or 1 pair (2 gens) + 3 others?
        # For k=5, we need 5 gens.
        # Let's try: G_c1, G_c1^-1, G_c2, G_c2^-1, H
        
        # Check H: x -> (x+1)/(x-1). a=1, b=1, c=1, d=-1.
        # Fixed points: x^2 - 2x - 1 = 0. Disc 8.
        if is_quadratic_residue(8, q):
            return None # H is not a derangement, try other H?
            # For now, if H fails, we might need a different 5th generator.
            # Let's just return None if H is not a derangement.
        
        c1 = candidates[0]
        c2 = candidates[1]
        
        # Define generators as (a,b,c,d)
        # G_c: x -> (x+c)/x = (1*x + c) / (1*x + 0). a=1, b=c, c=1, d=0.
        gen_c1 = (1, c1, 1, 0)
        gen_c1_inv = (1, 0, 1, -c1) # Inverse of (1, c, 1, 0) is (1, 0, 1, -c)?
        # Let's verify inverse.
        # M = [[1, c], [1, 0]]. Det = -c.
        # M^-1 = 1/(-c) [[0, -c], [-1, 1]] = [[0, 1], [1/c, -1/c]].
        # So a=0, b=1, c=1/c, d=-1/c.
        # Let's just compute inverse properly.
        
        def inv_matrix(a, b, c, d, q):
            det = (a*d - b*c) % q
            det_inv = modinv(det, q)
            if det_inv is None:
                return None
            a_inv = d * det_inv % q
            b_inv = -b * det_inv % q
            c_inv = -c * det_inv % q
            d_inv = a * det_inv % q
            return (a_inv, b_inv, c_inv, d_inv)
        
        g1 = gen_c1
        g1_inv = inv_matrix(*g1, q)
        g2 = (1, c2, 1, 0)
        g2_inv = inv_matrix(*g2, q)
        
        # H: x -> (x+1)/(x-1). a=1, b=1, c=1, d=-1.
        h = (1, 1, 1, q-1)
        
        if g1_inv is None or g2_inv is None:
            return None
            
        gens = [g1, g1_inv, g2, g2_inv, h]
        
        # Build graph
        adj = [set() for _ in range(n)]
        is_simple = True
        
        for i in range(n):
            x = i if i < q else q
            for gen in gens:
                img = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                if img is None:
                    is_simple = False
                    break
                if img == i:
                    is_simple = False
                    break
                adj[i].add(img)
                adj[img].add(i)
            if not is_simple:
                break
        
        if not is_simple:
            return None
            
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return None
                
        # Check connectivity
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if count != n:
            return None
            
        # Check girth
        if check_girth(adj, n, g):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
        return None

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

    # 1. Get the L-U-W baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)
    
    # 2. Try structured PGL(2, q)
    q_min = max(2, n_min - 1)
    q_max = best_n - 2
    
    if q_max >= q_min:
        # Iterate q in increasing order
        for curr_q in range(q_min, q_max + 1):
            if time.time() - start_time > timeout:
                break
            if not is_prime_power(curr_q):
                continue
            
            # Only try odd q for now
            if curr_q % 2 == 0:
                continue
                
            res = try_structured_pgl(curr_q, k, g)
            if res:
                n, edges = res
                if n < best_n:
                    best_n = n
                    best_result = res
                    # Since we iterate q increasing, this is the smallest n found so far.
                    # We can return immediately if we are confident this is optimal or good enough.
                    # But maybe a larger q has a smaller n? No, n=q+1.
                    # So the first valid q gives the smallest n.
                    return best_result

    return best_result

# EVOLVE-BLOCK-END
