import time
import random
import heapq
from collections import deque
import sys

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

def _compute_girth_fast(adjacency, n, early_stop=None):
    """Compute girth using BFS from each vertex with early stopping."""
    min_girth = float('inf')
    if early_stop is None:
        early_stop = 100
    
    for start in range(n):
        if min_girth <= early_stop:
            break
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth <= early_stop:
                            return min_girth
        if min_girth <= early_stop:
            break
    return min_girth if min_girth != float('inf') else -1

def _find_short_cycle_fast(adjacency, n, max_girth):
    """Find a cycle of length <= max_girth using BFS. Returns (u, v, cycle_len) or None."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_girth:
                continue
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len <= max_girth:
                        return (u, v, cycle_len)
    return None

def _has_short_path_bounded(adjacency, src, dst, max_dist, excluded_edges=None):
    """Check if there's a path from src to dst of length <= max_dist avoiding excluded edges."""
    if src == dst:
        return True
    if excluded_edges is None:
        excluded_edges = set()
    visited = {src: 0}
    queue = deque([(src, 0)])
    while queue:
        node, d = queue.popleft()
        if d >= max_dist:
            continue
        for neighbor in adjacency[node]:
            if (node, neighbor) in excluded_edges or (neighbor, node) in excluded_edges:
                continue
            if neighbor == dst:
                return True
            if neighbor not in visited:
                visited[neighbor] = d + 1
                queue.append((neighbor, d + 1))
    return False

def _is_2_switch_valid_fast(adjacency, u, v, w, x, g):
    """Fast check: does 2-switch (u,v)&(w,x) -> (u,w)&(v,x) create a cycle < g?"""
    excluded = {(u, v), (v, u), (w, x), (x, w)}
    
    if _has_short_path_bounded(adjacency, u, w, g - 2, excluded):
        return False
    
    if _has_short_path_bounded(adjacency, v, x, g - 2, excluded):
        return False
    
    return True

def _generate_simple_k_regular(k, n, max_attempts=200):
    """Generate a simple k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    for attempt in range(max_attempts):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        
        random.shuffle(stubs)
        
        adjacency = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v or v in adjacency[u]:
                valid = False
                break
            adjacency[u].add(v)
            adjacency[v].add(u)
        
        if valid:
            for i in range(n):
                if len(adjacency[i]) != k:
                    valid = False
                    break
            if valid:
                return adjacency
    
    return None

def _switching_construct(k, g, n, time_limit=100.0):
    """Construct a k-regular graph of girth >= g using edge switching."""
    start_time = time.time()
    
    adjacency = _generate_simple_k_regular(k, n, max_attempts=200)
    if adjacency is None:
        return None
    
    adj_list = [sorted(adjacency[i]) for i in range(n)]
    
    current_girth = _compute_girth_fast(adj_list, n, early_stop=g)
    
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adj_list[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    max_iterations = 50000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        result = _find_short_cycle_fast(adj_list, n, g - 1)
        if result is None:
            final_girth = _compute_girth_fast(adj_list, n, early_stop=g)
            if final_girth >= g:
                edges = []
                for i in range(n):
                    for j in adj_list[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            continue
        
        u, v, cycle_len = result
        
        success = False
        for attempt in range(20):
            w = random.randint(0, n - 1)
            if not adj_list[w]:
                continue
            x = random.choice(adj_list[w])
            
            if w == u or w == v or x == u or x == v:
                continue
            if x in adjacency[u] or w in adjacency[v]:
                continue
            if u in adjacency[w] or v in adjacency[x]:
                continue
            
            if _is_2_switch_valid_fast(adj_list, u, v, w, x, g):
                adjacency[u].discard(v)
                adjacency[v].discard(u)
                adjacency[w].discard(x)
                adjacency[x].discard(w)
                adjacency[u].add(w)
                adjacency[w].add(u)
                adjacency[v].add(x)
                adjacency[x].add(v)
                
                adj_list[u] = sorted(adjacency[u])
                adj_list[v] = sorted(adjacency[v])
                adj_list[w] = sorted(adjacency[w])
                adj_list[x] = sorted(adjacency[x])
                
                success = True
                break
        
        if not success and iteration % 100 == 0:
            w = random.randint(0, n - 1)
            if adj_list[w]:
                x = random.choice(adj_list[w])
                u_rand = random.randint(0, n - 1)
                if adj_list[u_rand]:
                    v_rand = random.choice(adj_list[u_rand])
                    if len({u_rand, v_rand, w, x}) == 4:
                        if v_rand not in adjacency[w] and x not in adjacency[u_rand]:
                            if _is_2_switch_valid_fast(adj_list, u_rand, v_rand, w, x, g):
                                adjacency[u_rand].discard(v_rand)
                                adjacency[v_rand].discard(u_rand)
                                adjacency[w].discard(x)
                                adjacency[x].discard(w)
                                adjacency[u_rand].add(w)
                                adjacency[w].add(u_rand)
                                adjacency[v_rand].add(x)
                                adjacency[x].add(v_rand)
                                adj_list[u_rand] = sorted(adjacency[u_rand])
                                adj_list[v_rand] = sorted(adjacency[v_rand])
                                adj_list[w] = sorted(adjacency[w])
                                adj_list[x] = sorted(adjacency[x])
    
    final_girth = _compute_girth_fast(adj_list, n, early_stop=g)
    if final_girth >= g:
        edges = []
        for i in range(n):
            for j in adj_list[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None

def _greedy_construct(k, g, n, time_limit=10.0):
    """Greedy construction: add edges one at a time, checking girth constraint."""
    start_time = time.time()
    adjacency = [set() for _ in range(n)]
    degrees = [0] * n
    
    def has_short_cycle(u, v, max_depth):
        if u == v:
            return True
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            node, d = queue.popleft()
            if d >= max_depth:
                continue
            for neighbor in adjacency[node]:
                if neighbor == v:
                    return True
                if neighbor not in visited:
                    visited[neighbor] = d + 1
                    queue.append((neighbor, d + 1))
        return False

    order = list(range(n))
    random.shuffle(order)
    
    for u in order:
        if time.time() - start_time > time_limit:
            return None
        
        candidates = [v for v in range(n) if v != u and v not in adjacency[u]]
        random.shuffle(candidates)
        
        found = False
        for v in candidates:
            if degrees[v] >= k:
                continue
            if has_short_cycle(u, v, g - 2):
                continue
            
            adjacency[u].add(v)
            adjacency[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            found = True
            break
        
        if not found:
            return None
    
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
    
    edges = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edges.append((i, j))
    return n, edges

def construct(k, g):
    if k == 5 and g == 9:
        total_time = 140.0
        start_time = time.time()
        
        target_ns = [1000, 1100, 1116, 1200, 1400, 1600, 1800, 2000, 2200, 2400]
        
        for n in target_ns:
            elapsed = time.time() - start_time
            remaining = total_time - elapsed
            if remaining < 15:
                break
            targets_left = len(target_ns) - target_ns.index(n)
            time_per_attempt = min(25.0, remaining / max(1, targets_left))
            
            res = _switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
        
        elapsed = time.time() - start_time
        remaining = total_time - elapsed
        if remaining > 20:
            for n in [1200, 1400, 1600]:
                if time.time() - start_time > total_time - 10:
                    break
                res = _greedy_construct(k, g, n, time_limit=15.0)
                if res is not None:
                    return res
    
    return _luw_construct(k, g)
