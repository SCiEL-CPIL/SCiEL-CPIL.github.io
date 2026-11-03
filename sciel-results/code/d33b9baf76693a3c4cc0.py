import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    # Moore bound
    if g % 2 == 0:
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        if k == 2:
            moore = g
        else:
            h = (g - 1) // 2
            moore = 1 + k * ((k - 1) ** h - 1) // (k - 2) if k > 2 else g

    def verify_girth_fast(n, adj, min_girth):
        if n == 0:
            return True
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check girth
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def generate_random_regular(n, k):
        if n * k % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(n * k // 2):
            u, v = stubs[2 * i], stubs[2 * i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not valid:
            return None
        return n, adj

    def find_short_cycle(n, adj, min_girth):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return (u, v)
        return None

    def try_repair(n, adj, u, v, k):
        neighbors_u = list(adj[u])
        neighbors_v = list(adj[v])
        random.shuffle(neighbors_u)
        random.shuffle(neighbors_v)
        
        for u_prime in neighbors_u:
            if u_prime == v:
                continue
            for v_prime in neighbors_v:
                if v_prime == u:
                    continue
                if u_prime == v_prime:
                    continue
                if v_prime in adj[u]:
                    continue
                if v in adj[u_prime]:
                    continue
                
                # Swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[u_prime].remove(v_prime)
                adj[v_prime].remove(u_prime)
                
                adj[u].add(v_prime)
                adj[v_prime].add(u)
                adj[u_prime].add(v)
                adj[v].add(u_prime)
                
                return True
        return False

    def luw_graph(k, g):
        q = k
        if (g - 5) % 2 == 1:
            d = g - 5
        else:
            d = g - 4
        if d < 3:
            d = 3

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    best_n = None
    best_edges = None

    # 1. LUW Baseline
    if time_left() > 5:
        try:
            n_luw, edges_luw = luw_graph(k, g)
            # Quick check: LUW is known to be valid.
            # We can skip full verify if we trust it, but let's do it to be safe.
            # Building adj for verify
            adj_luw = [set() for _ in range(n_luw)]
            for u, v in edges_luw:
                adj_luw[u].add(v)
                adj_luw[v].add(u)
            if verify_girth_fast(n_luw, adj_luw, g):
                best_n = n_luw
                best_edges = edges_luw
        except Exception:
            pass

    # 2. Random Search + Repair
    # Try to beat best_n
    # If best_n is None, we have no target, so we try to find any.
    # If best_n is 3125, we try < 3125.
    
    max_n_to_try = best_n if best_n else 3000
    # We only try if we can potentially beat it.
    # If best_n is small (e.g. 100), we don't search.
    
    # Start from Moore bound
    start_n = max(moore + 10, 100)
    
    if best_n is None or best_n > start_n:
        step = 50
        for n_try in range(start_n, max_n_to_try, step):
            if time_left() < 10:
                break
            
            # Try a few seeds
            for attempt in range(3):
                if time_left() < 5:
                    break
                
                res = generate_random_regular(n_try, k)
                if res is None:
                    continue
                
                n_r, adj_r = res
                
                # Repair
                for _ in range(500):
                    if time_left() < 5:
                        break
                    
                    cycle_edge = find_short_cycle(n_r, adj_r, g)
                    if cycle_edge is None:
                        # Verify
                        if verify_girth_fast(n_r, adj_r, g):
                            edges_r = [(u, v) for u in range(n_r) for v in adj_r[u] if u < v]
                            if best_n is None or n_r < best_n:
                                best_n = n_r
                                best_edges = edges_r
                            break # Success, break repair loop
                    
                    u, v = cycle_edge
                    if not try_repair(n_r, adj_r, u, v, k):
                        break # Repair failed, try new random graph
                
                if best_n is not None and best_n == n_r:
                    break # Found a valid graph at this N, break attempt loop
            
            if best_n is not None and best_n == n_try:
                break # Found a valid graph at this N, break N loop

    if best_n is None:
        # Fallback
        try:
            n_luw, edges_luw = luw_graph(k, g)
            return n_luw, edges_luw
        except Exception:
            pass

    return best_n, best_edges
