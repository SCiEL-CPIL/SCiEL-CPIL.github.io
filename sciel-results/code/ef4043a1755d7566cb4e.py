import time
import random
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Fallback: LUW construction
    def luw_fallback():
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

    # LUW is our guaranteed baseline
    n_luw, edges_luw = luw_fallback()
    best_n = n_luw
    best_edges = edges_luw
    
    # Try PSL(2, q) Cayley graphs for k=5
    if k == 5:
        candidate_qs = [11, 13, 17, 19, 23, 29, 31]
        
        for q in candidate_qs:
            if time.time() > deadline:
                break
            n = q * (q * q - 1) // 2
            if n >= best_n:
                continue
                
            # Generate PSL(2, q)
            try:
                group, idx_map, inverses, orders, involutions, norm_func = _generate_psl2(q)
            except Exception:
                continue
                
            if not involutions:
                continue
            
            # Search for 5 generators: 2 pairs + 1 involution
            found = False
            attempts = 0
            max_attempts = 5000 # Increase attempts for better chance
            
            while attempts < max_attempts:
                if time.time() > deadline:
                    break
                attempts += 1
                
                non_inv = [x for x in group if orders[x] != 2 and x != (1,0,0,1)]
                if not non_inv:
                    break
                
                s1 = random.choice(non_inv)
                s2 = random.choice(non_inv)
                s3 = random.choice(involutions)
                
                s1_inv = inverses[s1]
                s2_inv = inverses[s2]
                
                candidates = [s1, s1_inv, s2, s2_inv, s3]
                
                if len(set(candidates)) != 5:
                    continue
                    
                if _check_girth_psl(group, idx_map, candidates, g, q, norm_func):
                    edges = []
                    for i, g_elem in enumerate(group):
                        for s in candidates:
                            neighbor = _mul_psl(g_elem, s, q)
                            neighbor = norm_func(neighbor)
                            j = idx_map[neighbor]
                            if i < j:
                                edges.append((i, j))
                    
                    if len(edges) == n * 5 // 2:
                        best_n = n
                        best_edges = edges
                        found = True
                        return best_n, best_edges
            
    return best_n, best_edges

def _generate_psl2(q):
    elements = []
    idx_map = {}
    
    def norm(m):
        a_, b_, c_, d_ = m
        for val in [a_, b_, c_, d_]:
            if val != 0:
                if val > (q - 1) // 2:
                    return ((q - a_) % q, (q - b_) % q, (q - c_) % q, (q - d_) % q)
                break
        return m

    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0:
                    if b == 0:
                        continue
                    if (b * c) % q != (q - 1) % q:
                        continue
                    d = 0
                else:
                    d = (1 + b * c) % q
                    a_inv = pow(a, q - 2, q)
                    d = (d * a_inv) % q
                
                if (a * d - b * c) % q != 1:
                    continue
                
                m = (a, b, c, d)
                m = norm(m)
                
                if m not in idx_map:
                    idx_map[m] = len(elements)
                    elements.append(m)
    
    inverses = {}
    orders = {}
    involutions = []
    identity = (1, 0, 0, 1)
    
    for g_elem in elements:
        a, b, c, d = g_elem
        inv = (d, (q - b) % q, (q - c) % q, a)
        inv = norm(inv)
        inverses[g_elem] = inv
        
        if g_elem == identity:
            orders[g_elem] = 1
            continue
        
        g2 = _mul_psl(g_elem, g_elem, q)
        g2 = norm(g2)
        if g2 == identity:
            orders[g_elem] = 2
            involutions.append(g_elem)
        else:
            orders[g_elem] = -1
            
    return elements, idx_map, inverses, orders, involutions, norm

def _mul_psl(m1, m2, q):
    a1, b1, c1, d1 = m1
    a2, b2, c2, d2 = m2
    a = (a1 * a2 + b1 * c2) % q
    b = (a1 * b2 + b1 * d2) % q
    c = (c1 * a2 + d1 * c2) % q
    d = (c1 * b2 + d1 * d2) % q
    return (a, b, c, d)

def _check_girth_psl(group, idx_map, generators, g, q, norm_func):
    identity = (1, 0, 0, 1)
    dist = {identity: 0}
    parent = {identity: None}
    queue = collections.deque([identity])
    
    while queue:
        v = queue.popleft()
        if dist[v] >= g // 2:
            continue
            
        for s in generators:
            u = _mul_psl(v, s, q)
            u = norm_func(u)
            
            if u not in dist:
                dist[u] = dist[v] + 1
                parent[u] = v
                queue.append(u)
            else:
                if parent[u] == v or parent[v] == u:
                    continue
                cycle_len = dist[v] + 1 + dist[u]
                if cycle_len < g:
                    return False
                    
    return True
