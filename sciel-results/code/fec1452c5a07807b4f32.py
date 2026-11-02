import time
import random
import numpy as np
from collections import deque

_DEADLINE = 110.0

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    if k != 4 or g != 10:
        return luw_construct(k, g)

    t0 = time.time()
    
    # Moore bound for (4, 10) is 258.
    # Known smallest is 320.
    # We try to find a graph with n < 320.
    
    def gen_random_4reg(n):
        while True:
            stubs = list(range(n * 4))
            random.shuffle(stubs)
            edges = []
            valid = True
            seen = set()
            for i in range(0, len(stubs), 2):
                u = stubs[i] // 4
                v = stubs[i + 1] // 4
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append(key)
            if valid:
                return edges

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            adj[i].sort()
        return adj

    def check_girth_fast(adj, n, g):
        """
        Returns min_cycle_length.
        Uses BFS from each vertex.
        """
        min_cycle = g
        half_g = g // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif dist[v] > dist[u]:
                        # Back edge
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < g:
                                return min_cycle # Early exit if we find a short cycle
                            
        return min_cycle

    def optimize_graph(n, time_limit):
        """
        Tries to find a 4-regular graph of girth >= g with n vertices.
        Returns (n, edges) if successful, else None.
        """
        edges = gen_random_4reg(n)
        adj = build_adj(edges, n)
        
        # Initial check
        min_c = check_girth_fast(adj, n, g)
        if min_c >= g:
            return n, edges
        
        # SA to fix girth
        T = 2.0
        cooling = 0.999
        t_start = time.time()
        
        # Precompute edge set for fast lookup? 
        # For n=320, set lookup is fast enough.
        
        def do_swap(u, v, x, y, adj):
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            # Keep sorted for consistency, though not strictly needed for set logic
            # But check_girth uses adj[u] iteration. Sorting helps determinism.
            adj[u].sort()
            adj[v].sort()
            adj[x].sort()
            adj[y].sort()

        def undo_swap(u, v, x, y, adj):
            adj[u].remove(x)
            adj[x].remove(u)
            adj[v].remove(y)
            adj[y].remove(v)
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
            adj[u].sort()
            adj[v].sort()
            adj[x].sort()
            adj[y].sort()

        current_cost = g - min_c # Cost is g - min_cycle if min_cycle < g, else 0.
        # Actually, let's define cost as max(0, g - min_cycle).
        # If min_cycle < g, cost = g - min_cycle.
        
        while time.time() - t_start < time_limit:
            if current_cost == 0:
                final_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            final_edges.append((i, j))
                return n, final_edges

            # Generate move
            u = random.randint(0, n - 1)
            if not adj[u]: continue
            v = random.choice(adj[u])
            
            x = random.randint(0, n - 1)
            if not adj[x]: continue
            y = random.choice(adj[x])

            if u == v or x == y: continue
            if len({u, v, x, y}) < 4: continue
            if x in adj[u] or y in adj[v]: continue

            do_swap(u, v, x, y, adj)
            new_min_c = check_girth_fast(adj, n, g)
            new_cost = max(0, g - new_min_c)
            
            delta = new_cost - current_cost
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / T)):
                current_cost = new_cost
            else:
                undo_swap(u, v, x, y, adj)
            
            T *= cooling
            if T < 0.05: T = 0.05

        return None

    # Strategy: Try n from 319 down to 258
    # Allocate time based on remaining budget.
    # If we find one, return it.
    
    # We know 320 works.
    # Try 319, 318, etc.
    
    # To avoid spending too much time on unlikely candidates,
    # we give each candidate a small slice of time.
    
    remaining_time = _DEADLINE - (time.time() - t0)
    # Try a few candidates
    candidates = list(range(319, 257, -1))
    
    # We might not have time for all.
    # Let's try 319, 318, 317 with 10s each.
    # If we have more time, try more.
    
    # Actually, just try 319 with a generous budget, then 318, etc.
    # But if 319 fails after 10s, it's unlikely 318 will succeed quickly.
    
    # Let's try 319 for 20s.
    # If it works, great.
    # If not, try 318 for 10s.
    # If not, return LUW.
    
    # To be safe, ensure we return LUW if we run out of time.
    
    if remaining_time > 20:
        res = optimize_graph(319, 20.0)
        if res:
            return res
        
    if time.time() - t0 > _DEADLINE - 10:
        return luw_construct(k, g)
        
    if time.time() - t0 < _DEADLINE - 10:
        res = optimize_graph(318, 10.0)
        if res:
            return res
            
    return luw_construct(k, g)
