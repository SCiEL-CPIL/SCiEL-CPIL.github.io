import time
import random
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    # (Kept from seed to ensure validity if search fails)
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
    fallback_n, fallback_edges = len(order), edges

    # Search for a smaller circulant graph
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def check_girth(n, steps):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for s in steps:
                u = (i + s) % n
                v = (i - s) % n
                if u == i or v == i:
                    return False # Loop
                if u in adj[i] or v in adj[i]:
                    return False # Repeated edge (if s=0 or 2s=0)
                adj[i].append(u)
                adj[i].append(v)
        
        # Check for simple graph (no multi-edges)
        # Since it's circulant, we just need to ensure steps are distinct mod n
        # and no step is 0 or n/2 (which would be a double edge if n is even and s=n/2? No, s and -s are same if 2s=0)
        # If n is even and s = n/2, then i+s = i-s. So we add the same neighbor twice.
        # So we must ensure 2*s != 0 mod n.
        
        # Girth check via BFS from vertex 0 (and maybe others if time permits)
        # For circulant graphs, if girth is small, it's likely visible from 0.
        # But to be safe, we should check all. However, for speed, we check 0 first.
        
        def bfs_girth(start):
            dist = [-1] * n
            prev = [-1] * n
            dist[start] = 0
            q = deque([start])
            min_girth = float('inf')
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        q.append(v)
                    elif prev[u] != v:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        min_girth = min(min_girth, cycle_len)
            return min_girth

        g0 = bfs_girth(0)
        if g0 < g:
            return False
        # Optional: check a few more vertices if time allows
        return True

    # Search space
    # n must be at least Moore bound
    moore_bound = 2 * (2**(g//2) - 1) // (k-2) if k > 2 else 0
    if k == 3:
        moore_bound = 2 * (2**(g//2 - 1) - 1) // 1 # 2 * (2^7 - 1) = 510 for g=16
    else:
        moore_bound = 2 * sum((k-1)**i for i in range(g//2))
        
    # Adjust Moore bound calculation for k=3, g=16
    # n >= 2 * sum_{i=0}^{7} 2^i = 2 * (2^8 - 1) = 510
    
    found = False
    best_n = fallback_n
    best_edges = fallback_edges
    
    # Try small n first
    for n in range(moore_bound, min(moore_bound + 500, 2000)):
        if time.time() > deadline:
            break
        # Generate steps
        # Steps must be distinct and not 0 or n/2
        valid_steps = [s for s in range(1, n//2 + 1) if 2*s % n != 0]
        if len(valid_steps) < k:
            continue
        # We need to choose k steps.
        # To keep it simple, we can iterate combinations.
        # For k=3, combinations of 3 from ~n/2 options.
        # This is too many.
        # Instead, we can use a greedy/random approach for steps.
        # Or restrict to "consecutive" steps or "sparse" steps.
        # Let's try random sampling of steps.
        
        # Actually, for circulant graphs, the girth is determined by the smallest solution to
        # sum of steps = 0.
        # We can just randomize steps and check.
        
        for _ in range(100): # 100 random attempts per n
            if time.time() > deadline:
                break
            steps = random.sample(valid_steps, k)
            if check_girth(n, steps):
                # Construct edges
                edges = []
                for i in range(n):
                    for s in steps:
                        u = (i + s) % n
                        if i < u:
                            edges.append((i, u))
                return n, edges

    return fallback_n, fallback_edges
