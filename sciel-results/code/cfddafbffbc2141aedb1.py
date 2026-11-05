import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

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

    def generate_random_regular(n, k, seed=42):
        """Generate a random k-regular graph on n vertices using configuration model + fixups."""
        rng = random.Random(seed)
        if n * k % 2 != 0:
            return None
        # Build degree sequence
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        rng.shuffle(stubs)
        # Pair up stubs
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not valid:
            return None
        return [sorted(adj[i]) for i in range(n)]

    def generate_random_regular_bipartite(n_half, k, seed=42):
        """Generate a random k-regular bipartite graph with two parts of size n_half."""
        rng = random.Random(seed)
        adj_a = [set() for _ in range(n_half)]
        adj_b = [set() for _ in range(n_half)]
        # Each vertex in A connects to k vertices in B
        for i in range(n_half):
            targets = rng.sample(range(n_half), k)
            for j in targets:
                adj_a[i].add(j)
                adj_b[j].add(i)
        # Verify regularity
        for i in range(n_half):
            if len(adj_a[i]) != k or len(adj_b[i]) != k:
                return None
        return [sorted(adj_a[i]) for i in range(n_half)] + [sorted(n_half + j for j in adj_b[i]) for i in range(n_half)]

    def generate_random_regular_bipartite_v2(n_half, k, seed=42):
        """Better: generate k-regular bipartite via random permutation composition."""
        rng = random.Random(seed)
        # Generate k random permutations of range(n_half)
        perms = []
        for _ in range(k):
            p = list(range(n_half))
            rng.shuffle(p)
            perms.append(p)
        adj_a = [set() for _ in range(n_half)]
        adj_b = [set() for _ in range(n_half)]
        for i in range(n_half):
            for perm in perms:
                j = perm[i]
                adj_a[i].add(j)
                adj_b[j].add(i)
        for i in range(n_half):
            if len(adj_a[i]) != k or len(adj_b[i]) != k:
                return None
        return [sorted(adj_a[i]) for i in range(n_half)] + [sorted(n_half + j for j in adj_b[i]) for i in range(n_half)]

    def generate_random_regular_config(n, k, seed=42):
        """Configuration model with retry."""
        rng = random.Random(seed)
        for attempt in range(50):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            rng.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                return [sorted(adj[i]) for i in range(n)]
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

    # Start with LUW construction
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)

    # Try to improve via random regular graph search
    # Strategy: try n values from Moore bound up to best_n - 1
    # For each n, try random k-regular graphs and check girth
    
    # For bipartite (even girth), use bipartite constructions
    # For odd girth, use general constructions
    
    max_search_n = min(best_n - 1, 2000)  # Cap to keep runtime bounded
    search_start = n_min
    
    # Determine if we should use bipartite or general
    use_bipartite = (g % 2 == 0)
    
    # Number of attempts per n value
    attempts_per_n = 20
    
    for n_try in range(search_start, max_search_n + 1):
        if time.time() > deadline - 5.0:
            break
        if n_try >= best_n:
            break
        if use_bipartite:
            if n_try % 2 != 0:
                continue
            n_half = n_try // 2
            if n_half * k % 2 != 0:
                continue
            for attempt in range(attempts_per_n):
                if time.time() > deadline - 5.0:
                    break
                adj = generate_random_regular_bipartite_v2(n_half, k, seed=attempt * 1000 + n_try)
                if adj is None:
                    continue
                if not is_connected(adj, n_try):
                    continue
                if check_girth_nb(adj, n_try, g):
                    edges = []
                    for i in range(n_try):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    best_n = n_try
                    best_result = (n_try, edges)
                    break
        else:
            if n_try * k % 2 != 0:
                continue
            for attempt in range(attempts_per_n):
                if time.time() > deadline - 5.0:
                    break
                adj = generate_random_regular_config(n_try, k, seed=attempt * 1000 + n_try)
                if adj is None:
                    continue
                if not is_connected(adj, n_try):
                    continue
                if check_girth_nb(adj, n_try, g):
                    edges = []
                    for i in range(n_try):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    best_n = n_try
                    best_result = (n_try, edges)
                    break

    return best_result

# EVOLVE-BLOCK-END
