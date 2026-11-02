import time
import random
import sys
from collections import deque

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _generate_random_k_regular(n, k):
    if (n * k) % 2 != 0:
        return None
    
    while True:
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
            
        if valid:
            return adj

def _compute_girth(adj, k, g_max):
    n = len(adj)
    min_girth = g_max
    for start in range(n):
        visited = {}
        queue = deque()
        for neighbor in adj[start]:
            visited[(neighbor, start)] = 1
            queue.append((neighbor, start, 1))
        while queue:
            vertex, prev, depth = queue.popleft()
            if start in adj[vertex] and depth + 1 < min_girth:
                min_girth = depth + 1
                if min_girth <= g_max:
                    return min_girth
            for neighbor in adj[vertex]:
                if neighbor == prev:
                    continue
                state = (neighbor, vertex)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((neighbor, vertex, depth + 1))
    return min_girth

def _has_short_path(adj, u, v, max_depth):
    if u == v:
        return True
    visited = {u: 0}
    q = deque([(u, 0)])
    while q:
        node, d = q.popleft()
        if d >= max_depth:
            continue
        for nb in adj[node]:
            if nb == v:
                return True
            if nb not in visited:
                visited[nb] = d + 1
                q.append((nb, d + 1))
    return False

def _find_short_cycle(adj, g):
    n = len(adj)
    # Start from a random vertex to diversify
    start = random.randint(0, n - 1)
    visited = {start: (None, 0)}
    q = deque([(start, None, 0)])
    while q:
        u, prev, d = q.popleft()
        if d >= g - 1:
            continue
        for v in adj[u]:
            if v == prev:
                continue
            if v == start and d + 1 < g:
                # Found cycle
                path = [v, u]
                cur = u
                while cur != start:
                    cur = visited[cur][0]
                    path.append(cur)
                path.reverse()
                return path
            if v not in visited:
                visited[v] = (u, d + 1)
                q.append((v, u, d + 1))
    return None

def _switching_construct_v2(k, g, n, time_limit):
    start_time = time.time()
    adj = _generate_random_k_regular(n, k)
    if adj is None:
        return None
    
    while time.time() - start_time < time_limit:
        cycle = _find_short_cycle(adj, g)
        if cycle is None:
            if _compute_girth(adj, k, g) >= g:
                edges_list = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges_list.append((i, j))
                return n, edges_list
            else:
                break
                
        # Try to fix an edge in the cycle
        success = False
        for idx in range(len(cycle)):
            u = cycle[idx]
            v = cycle[(idx + 1) % len(cycle)]
            
            # Try 20 random swaps
            for _ in range(20):
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y or x in (u, v) or y in (u, v):
                    continue
                if y not in adj[x]:
                    continue
                if (x, y) == (u, v) or (x, y) == (v, u):
                    continue
                
                # Try switch
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                max_depth = g - 2
                reject = False
                if _has_short_path(adj, u, x, max_depth):
                    reject = True
                elif _has_short_path(adj, v, y, max_depth):
                    reject = True
                
                if reject:
                    # Try other pairing
                    adj[u].add(v)
                    adj[v].add(u)
                    adj[x].add(y)
                    adj[y].add(x)
                    
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    reject2 = False
                    if _has_short_path(adj, u, y, max_depth):
                        reject2 = True
                    elif _has_short_path(adj, v, x, max_depth):
                        reject2 = True
                        
                    if reject2:
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[x].add(y)
                        adj[y].add(x)
                    else:
                        adj[u].add(y)
                        adj[y].add(u)
                        adj[v].add(x)
                        adj[x].add(v)
                        success = True
                        break
                else:
                    adj[u].add(x)
                    adj[x].add(u)
                    adj[v].add(y)
                    adj[y].add(v)
                    success = True
                    break
                    
            if success:
                break
                
        if not success:
            # If we couldn't fix any edge in this cycle, try again
            continue

    return None

def construct(k, g):
    start_time = time.time()
    time_limit = 145.0
    
    if k == 5 and g == 9:
        target_ns = [450, 500, 600, 800, 1000, 1200]
        
        for n in target_ns:
            if time.time() - start_time > time_limit:
                break
            remaining = time_limit - (time.time() - start_time)
            if remaining < 10.0:
                break
            t_alloc = min(10.0, remaining)
            result = _switching_construct_v2(k, g, n, t_alloc)
            if result is not None:
                return result
                
    return _luw_construct(k, g)
