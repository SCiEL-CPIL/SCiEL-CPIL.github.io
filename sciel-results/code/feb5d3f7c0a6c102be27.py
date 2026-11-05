import random
import time
import math

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Collect distinct edge incidence vectors for non-backtracking closed walks of length < g
    walks_vecs = set()
    for start_v in range(n_base):
        queue = []
        for next_v, e_id in adj[start_v]:
            counts = [0] * 15
            counts[e_id] = 1
            queue.append((next_v, e_id, 1, tuple(counts)))
            
        while queue:
            curr_v, prev_e, length, counts = queue.pop(0)
            if length >= g:
                continue
            if curr_v == start_v and length >= 3:
                walks_vecs.add(counts)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                new_counts = list(counts)
                new_counts[e_id] += 1
                queue.append((next_v, e_id, length + 1, tuple(new_counts)))
                
    # Convert to list for faster iteration
    walks_list = list(walks_vecs)
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = float('inf')
    best_edges = None
    best_m = None
    
    for m in range(52, 2000):
        if time.time() > deadline - 15.0:
            break
            
        candidate_n = 10 * m
        
        found = False
        # Adaptive tries
        time_left = deadline - time.time()
        max_tries = min(20000, max(1000, int(time_left / 0.001)))
        
        for trial in range(max_tries):
            if time.time() > deadline - 3.0:
                break
                
            volts = [random.randint(1, m-1) for _ in range(15)]
            
            # Quick gcd check
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
                if g_val == 1:
                    break
            if g_val != 1:
                continue
                
            valid = True
            for counts in walks_list:
                s = 0
                for i, c in enumerate(counts):
                    if c:
                        s += c * volts[i]
                if s % m == 0:
                    valid = False
                    break
                    
            if valid:
                found = True
                break
                
        if found:
            best_m = m
            best_n = candidate_n
            lift_edges = []
            for i, (u, v) in enumerate(edges):
                vol = volts[i]
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v * m + (t + vol) % m
                    lift_edges.append((u_lift, v_lift))
            best_edges = lift_edges
            break
            
    if best_edges is None:
        return _luw_fallback(k, g)
        
    return best_n, best_edges

def _luw_fallback(k, g):
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")
