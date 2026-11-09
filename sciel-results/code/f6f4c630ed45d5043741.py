import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Fallback for general case
    if not (k == 3 and g == 16):
        return _seed_construct(k, g)

    # Specific optimization for 3-regular, girth 16
    # Try to find a graph with N=936 (known to exist)
    # If found, return it.
    # If not, try smaller N.
    # If not found in time, return LUW.
    
    # We'll try a few N values.
    # 936 is the target.
    # We'll also try 512 (lower bound) just in case.
    
    candidate_ns = [936, 934, 932, 930, 928, 926, 924, 922, 920, 900, 800, 700, 600, 512]
    
    for N in candidate_ns:
        if time.time() > deadline - 10: # Leave 10s for fallback
            break
        if N < 512 or N % 2 != 0:
            continue
            
        # Try to find a graph of size N with girth >= 16
        result = _search_graph(N, k, g, deadline)
        if result:
            return result

    # Fallback
    return _seed_construct(k, g)

def _seed_construct(k, g):
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

def _search_graph(N, k, g, deadline):
    # Local search to maximize girth
    # Start with random k-regular graph
    
    def generate_random_regular(N, k):
        stubs = []
        for i in range(N):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(N)]
        for i in range(0, N*k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return generate_random_regular(N, k)
            if v in adj[u]:
                return generate_random_regular(N, k)
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def get_girth(adj):
        min_girth = float('inf')
        for u in range(N):
            dist = [-1] * N
            parent = [-1] * N
            dist[u] = 0
            queue = [u]
            while queue:
                v = queue.pop(0)
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v] and parent[w] != v:
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    adj = generate_random_regular(N, k)
    current_girth = get_girth(adj)
    
    if current_girth >= g:
        edges = []
        for u in range(N):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return N, edges

    # Local search
    # Use simulated annealing or simple hill climbing
    # We'll do a fixed number of steps
    steps = 5000
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Pick two random edges
        edge_list = []
        for u in range(N):
            for v in adj[u]:
                if u < v:
                    edge_list.append((u, v))
                    
        if len(edge_list) < 2: break
        e1, e2 = random.sample(edge_list, 2)
        a, b = e1
        c, d = e2
        
        # Try swap
        valid_swap = False
        new_adj = None
        for new_e1, new_e2 in [((a,c), (b,d)), ((a,d), (b,c))]:
            u1, v1 = new_e1
            u2, v2 = new_e2
            if u1 == v1 or u2 == v2: continue
            if v1 in adj[u1] or v2 in adj[u2]: continue
            if u1 in adj[v1] or u2 in adj[v2]: continue
            
            # Perform swap on a copy? Too slow.
            # Do it in place and revert if bad.
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            
            adj[a].append(c)
            adj[c].append(a)
            adj[b].append(d)
            adj[d].append(b)
            
            new_girth = get_girth(adj)
            if new_girth >= g:
                edges = []
                for u in range(N):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return N, edges
            elif new_girth > current_girth:
                current_girth = new_girth
            else:
                # Revert
                adj[a].remove(c)
                adj[c].remove(a)
                adj[b].remove(d)
                adj[d].remove(b)
                
                adj[a].append(b)
                adj[b].append(a)
                adj[c].append(d)
                adj[d].append(c)
            valid_swap = True
            break
            
    return None
