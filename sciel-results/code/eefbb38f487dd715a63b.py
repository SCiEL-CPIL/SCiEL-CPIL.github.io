import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def luw_construction():
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

    def check_girth_fast(adj, n, g):
        """Check if graph has girth >= g using BFS from each vertex up to depth g//2."""
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif dist[v] != dist[u] + 1:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False, cycle_len
        return True, g

    def count_short_cycles_fast(adj, n):
        """Count triangles and 4-cycles as fast penalty."""
        count = 0
        adj_set = [set(a) for a in adj]
        for u in range(n):
            for i in range(len(adj[u])):
                for j in range(i + 1, len(adj[u])):
                    if adj[u][j] in adj_set[adj[u][i]]:
                        count += 1
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj_set[u] & adj_set[v]
                    count += len(common)
        return count

    def gen_random_k_reg(n, k):
        for attempt in range(500):
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid and all(len(a) == k for a in adj):
                return adj
        return None

    def optimize(n, k, g, time_limit):
        adj = gen_random_k_reg(n, k)
        if not adj:
            return None
        
        adj_set = [set(a) for a in adj]
        current_penalty = count_short_cycles_fast(adj, n)
        end_time = time.time() + time_limit
        iter_count = 0
        
        while time.time() < end_time:
            u = random.randint(0, n - 1)
            if not adj[u]:
                continue
            v = random.choice(adj[u])
            
            w = random.randint(0, n - 1)
            if w in (u, v) or not adj[w]:
                continue
            x = random.choice(adj[w])
            if x in (u, v) or w == x:
                continue
            
            if w in adj_set[u] or x in adj_set[v]:
                continue
            
            adj[u].remove(v)
            adj_set[u].discard(v)
            adj[v].remove(u)
            adj_set[v].discard(u)
            adj[w].remove(x)
            adj_set[w].discard(x)
            adj[x].remove(w)
            adj_set[x].discard(w)
            
            adj[u].append(w)
            adj_set[u].add(w)
            adj[w].append(u)
            adj_set[w].add(u)
            adj[v].append(x)
            adj_set[v].add(x)
            adj[x].append(v)
            adj_set[x].add(v)
            
            new_penalty = count_short_cycles_fast(adj, n)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                iter_count += 1
                if iter_count % 500 == 0:
                    ok, actual_girth = check_girth_fast(adj, n, g)
                    if ok:
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return (n, edges)
            else:
                adj[u].remove(w)
                adj_set[u].discard(w)
                adj[w].remove(u)
                adj_set[w].discard(u)
                adj[v].remove(x)
                adj_set[v].discard(x)
                adj[x].remove(v)
                adj_set[x].discard(v)
                
                adj[u].append(v)
                adj_set[u].add(v)
                adj[v].append(u)
                adj_set[v].add(u)
                adj[w].append(x)
                adj_set[w].add(x)
                adj[x].append(w)
                adj_set[x].add(w)
        
        ok, _ = check_girth_fast(adj, n, g)
        if ok:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return (n, edges)
        return None

    def moore_bound(k, g):
        if g % 2 == 0:
            d = g // 2
            return k + k * (k - 1) * sum((k - 1) ** i for i in range(d - 1))
        else:
            d = (g - 1) // 2
            return k + k * sum((k - 1) ** i for i in range(d))

    lb = moore_bound(k, g)
    
    candidates = []
    for mult in [1.0, 1.1, 1.2, 1.5, 2.0, 3.0]:
        n = int(lb * mult)
        if n % 2 == 0 and n * k % 2 == 0:
            candidates.append(n)
    
    candidates.extend([500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1500])
    candidates = sorted(set(candidates))
    
    best_result = None
    
    for n in candidates:
        if time.time() > deadline - 15:
            break
        
        time_budget = min(15, deadline - time.time() - 10)
        if time_budget < 3:
            break
        
        result = optimize(n, k, g, time_budget)
        if result:
            if best_result is None or result[0] < best_result[0]:
                best_result = result
                if best_result[0] <= lb + 5:
                    break
    
    if best_result:
        return best_result
    
    return luw_construction()
