import itertools
import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
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
        # Check if n = p^e for some prime p and e >= 2
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
        # Extended Euclidean algorithm
        def egcd(a, b):
            if b == 0:
                return (a, 1, 0)
            g, x1, y1 = egcd(b, a % b)
            return (g, y1, x1 - (a // b) * y1)
        g, x, _ = egcd(a % m, m)
        if g != 1:
            return None
        return x % m

    # We will search for q such that n = q+1 is small.
    # n must be at least the Moore bound.
    # Moore bound for k-regular, girth g:
    # If g is odd (g=2r+1): n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    # If g is even (g=2r): n >= 2 * sum_{i=0}^{r-1} (k-1)^i
    
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
    
    # We need q such that q+1 >= n_min.
    # So q >= n_min - 1.
    # Also q should be a prime power.
    
    # Generate candidate q values
    # We might need to go a bit higher than n_min - 1 because not all q work.
    # Let's try q from n_min - 1 up to some limit.
    # If n_min is large, this might be slow.
    # However, for small k, g, n_min is small.
    
    candidates_q = []
    # Start from max(2, n_min - 1)
    q_start = max(2, n_min - 1)
    # Limit search to reasonable q. If q is too large, n is large, but we want small n.
    # So we prefer small q.
    # We will check q in increasing order.
    
    def action_on_point(x, a, b, c, d, q):
        """
        Compute action of matrix [[a,b],[c,d]] on point x in projective line over Z_q.
        x is in 0..q-1 or infinity (represented by q).
        Returns the image point, or None if the action is undefined.
        """
        if x == q:  # infinity
            if c == 0:
                return q
            else:
                # a/c mod q
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

    def build_and_check(q, k, g):
        """
        Try to build a k-regular graph of girth >= g on q+1 vertices using
        random generators from PGL(2, q) acting on the projective line.
        Returns (n, edges) if successful, else None.
        """
        n = q + 1
        
        # Precompute: for each generator and each point, the image
        # This allows fast graph construction
        
        for attempt in range(500):
            if time.time() - start_time > timeout:
                return None
            
            # Generate k random invertible matrices
            gens = []
            valid = True
            for _ in range(k):
                while True:
                    a = random.randint(0, q - 1)
                    b = random.randint(0, q - 1)
                    c = random.randint(0, q - 1)
                    d = random.randint(0, q - 1)
                    det = (a * d - b * c) % q
                    if det != 0:
                        break
                gens.append((a, b, c, d))
            
            # Precompute images: images[i][j] = image of point i under generator j
            images = [[None] * k for _ in range(n)]
            valid = True
            for i in range(n):
                x = i if i < q else q
                for j in range(k):
                    img = action_on_point(x, gens[j][0], gens[j][1], gens[j][2], gens[j][3], q)
                    if img is None:
                        valid = False
                        break
                    images[i][j] = img
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Build adjacency list
            adj = [set() for _ in range(n)]
            is_simple = True
            
            for i in range(n):
                for j in range(k):
                    img = images[i][j]
                    if img == i:
                        # Loop - invalid
                        is_simple = False
                        break
                    adj[i].add(img)
                    adj[img].add(i)
                if not is_simple:
                    break
            
            if not is_simple:
                continue
            
            # Check degree
            for i in range(n):
                if len(adj[i]) != k:
                    is_simple = False
                    break
            
            if not is_simple:
                continue
            
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
                continue
            
            # Check girth using BFS from each vertex
            # For each u, find shortest path from u to any neighbor of u
            # (excluding the direct edge u->v, we want paths of length >= 2)
            # Actually, we want cycles, so we look for paths from u to a neighbor
            # of u that don't use the direct edge.
            
            # More efficient: for each u, BFS and track parent to avoid
            # going back along the same edge.
            
            min_cycle = float('inf')
            for u in range(n):
                # BFS from u
                dist = [-1] * n
                parent = [-1] * n
                dist[u] = 0
                bfs_queue = deque([u])
                
                while bfs_queue:
                    x = bfs_queue.popleft()
                    for y in adj[x]:
                        if dist[y] == -1:
                            dist[y] = dist[x] + 1
                            parent[y] = x
                            bfs_queue.append(y)
                        elif y != parent[x] and parent[y] != x:
                            # Found a cycle
                            # Cycle length = dist[x] + dist[y] + 1
                            cycle_len = dist[x] + dist[y] + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
                            # Early termination if we've already found a cycle
                            # shorter than g
                            if min_cycle < g:
                                break
                    if min_cycle < g:
                        break
                
                if min_cycle < g:
                    break
            
            if min_cycle >= g:
                # Valid graph
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        return None

    # Fallback: L-U-W construction from seed
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

    best_result = None
    best_n = float('inf')
    
    # Try Morgenstern for small q
    # We will try q up to a certain limit or until time runs out.
    # Since we want small n, we try small q first.
    
    # Determine search range
    q_start = max(2, n_min - 1)
    # If k is small, we can afford to search more q values
    if k <= 5:
        max_q_to_check = max(300, n_min + 100)
    elif k <= 10:
        max_q_to_check = max(200, n_min + 50)
    else:
        max_q_to_check = max(100, n_min + 20)
    
    for curr_q in range(q_start, max_q_to_check + 1):
        if time.time() - start_time > timeout:
            break
        
        # Check if curr_q is prime power
        if not is_prime_power(curr_q):
            continue
            
        res = build_and_check(curr_q, k, g)
        if res:
            n, edges = res
            if n < best_n:
                best_n = n
                best_result = res
        
        if time.time() - start_time > timeout:
            break

    if best_result:
        return best_result
    else:
        # Fallback to L-U-W
        return luw_construct(k, g)

# EVOLVE-BLOCK-END
