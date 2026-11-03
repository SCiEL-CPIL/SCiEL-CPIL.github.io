import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))

    def seed_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            rr, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rr]

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

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Certified exact girth check via non-backtracking BFS from every vertex."""
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def check_girth_circulant(n, S, g):
        """
        Vertex-transitive girth check for circulant graph C_n(S).
        BFS from vertex 0: if any non-tree edge creates a cycle < g, fail.
        """
        max_depth = (g - 1) // 2
        dist = [-1] * n
        parent = [-1] * n
        dist[0] = 0
        q = deque([0])

        while q:
            u = q.popleft()
            if dist[u] >= max_depth:
                continue
            for v in S:
                w = (u + v) % n
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    q.append(w)
                else:
                    if w == parent[u]:
                        continue
                    cycle_len = dist[u] + dist[w] + 1
                    if cycle_len < g:
                        return False
        return True

    def has_small_cycle(n, a, b):
        """
        Quick check for 3-cycles and 4-cycles in C_n({n/2, a, -a, b, -b}).
        Returns True if a small cycle is detected (candidate should be skipped).
        """
        h = n // 2
        S = {h, a, n-a, b, n-b}
        
        # Check 3-cycles: s1 + s2 + s3 = 0
        # Equivalent to s1 + s2 = -s3
        # Since S is symmetric, check if sum of two elements is in S (as -s3)
        # Actually, just check if any pair sums to an element in S (mod n)
        # But we need distinct edges? No, just existence.
        # 0 -> s1 -> s1+s2 -> 0 implies s1+s2 = -s3.
        # So s1+s2+s3 = 0.
        
        # Specific conditions for 3-cycles in this symmetric set:
        # 1. 2a = h (a = n/4) -> 0-h-a-0? No. 
        #    If a = n/4, then 2a = h. 
        #    Cycle: 0 -> h -> h+a = 3n/4 = -a -> 0. 
        #    Edges: (0,h), (h,-a), (-a,0). 
        #    h in S. -a-h = -a-n/2 = -3n/4 = n/4 = a in S. 0-(-a)=a in S. 
        #    So if a == n/4 or b == n/4, 3-cycle.
        if a == n // 4 or b == n // 4:
            return True
            
        # 2. a + b = h (a+b = n/2)
        #    Cycle: 0 -> a -> a+b=h -> 0.
        #    Edges: (0,a), (a,h), (h,0).
        #    a in S. h-a=b in S. h in S.
        if (a + b) % n == h or (a - b) % n == h or (b - a) % n == h:
            return True

        # 3. 2a = b or 2a = -b etc.
        #    If 2a = b, then 0 -> a -> 2a=b -> 0? No, need 3 steps.
        #    0 -> a -> a+b -> 0? a+a+b = 2a+b = b+b = 2b. Not 0.
        #    Check if any 3 elements sum to 0.
        #    h + a + (-a) = h != 0.
        #    a + (-a) + h = h != 0.
        #    a + b + (-a-b)? Is -a-b in S?
        #    If a+b = h, then -a-b = -h = h. In S. (Covered above)
        #    If a+b = a -> b=0 (No)
        #    If a+b = -a -> 2a+b=0.
        #    Let's just do a small loop for 3-cycle check for robustness.
        s_list = list(S)
        for i in range(len(s_list)):
            for j in range(i+1, len(s_list)):
                s3 = (-s_list[i] - s_list[j]) % n
                if s3 in S:
                    # Ensure it's not a trivial 2-cycle or loop
                    # 3-cycle requires distinct vertices? 
                    # 0, s1, s1+s2. 
                    # If s1=s2, then 2s1+s3=0.
                    # If s1=-s3, then 2s1=0 -> s1=h. 
                    # If s1=h, s2=a, s3=-a. h+a-a=h != 0.
                    # So if sum is 0, it's a 3-cycle.
                    return True
        return False

    def try_circulant_search(k, g, seed_n, deadline):
        """
        Search for a circulant k-regular graph with girth >= g and n < seed_n.
        """
        if k != 5:
            return None

        max_depth = (g - 1) // 2
        tree_size = 1 + k * sum((k - 1) ** i for i in range(max_depth))

        start_n = max(moore_bound, tree_size)
        if start_n % 2 != 0:
            start_n += 1
        start_n = max(start_n, 8)

        end_n = min(seed_n, start_n + 1000) # Search range

        for n in range(start_n, end_n, 2):
            if time.time() > deadline:
                break

            half_n = n // 2
            num_candidates = half_n - 1
            if num_candidates < 2:
                continue

            # Deterministic sweep for small n
            total_pairs = num_candidates * (num_candidates - 1) // 2
            
            if total_pairs <= 5000:
                # Full sweep
                for a in range(1, half_n):
                    if time.time() > deadline:
                        break
                    for b in range(a + 1, half_n):
                        if has_small_cycle(n, a, b):
                            continue
                        S = (half_n, a, n - a, b, n - b)
                        if len(set(S)) != 5:
                            continue
                        if check_girth_circulant(n, S, g):
                            edges = []
                            for i in range(n):
                                for s in S:
                                    j = (i + s) % n
                                    if i < j:
                                        edges.append((i, j))
                            return n, edges
            else:
                # Random sampling
                # Increase samples for larger n
                samples = min(5000, max(200, 500000 // n))
                for trial in range(samples):
                    if time.time() > deadline:
                        break
                    a = random.randint(1, half_n - 1)
                    b = random.randint(1, half_n - 1)
                    if a == b:
                        continue
                    if has_small_cycle(n, a, b):
                        continue
                    S = (half_n, a, n - a, b, n - b)
                    if len(set(S)) != 5:
                        continue
                    if check_girth_circulant(n, S, g):
                        edges = []
                        for i in range(n):
                            for s in S:
                                j = (i + s) % n
                                if i < j:
                                    edges.append((i, j))
                        return n, edges

        return None

    # Build the seed construction first
    seed_n, seed_edges = seed_construct(k, g)

    # Try circulant search
    result = try_circulant_search(k, g, seed_n, deadline)

    if result is not None:
        n, edges = result
        # Verify
        adj = build_adj(edges, n)
        if exact_girth_check(n, adj, g):
            return n, edges

    # Fallback
    return seed_n, seed_edges
