import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Generate LUW graph as fallback
    n_luw, edges_luw = _luw_graph(k, g)
    
    # 2. Try to find a smaller graph using voltage assignment on K_4
    if k == 3 and g == 16:
        # Target n=936, Base K_4 (4 vertices), Group Z_234
        result = _search_k4_voltage(k, g, 234, deadline)
        if result:
            n, edges = result
            if n < n_luw:
                return n, edges
                
    return n_luw, edges_luw

def _luw_graph(k, g):
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

def _search_k4_voltage(k, g, Q, deadline):
    # K_4 edges: 0-1, 0-2, 0-3, 1-2, 1-3, 2-3
    # Indices: 0:0-1, 1:0-2, 2:0-3, 3:1-2, 4:1-3, 5:2-3
    
    best_girth = 0
    best_voltages = None
    
    # Greedy local search
    current = [random.randint(0, Q-1) for _ in range(6)]
    
    for _ in range(20000):
        if time.time() > deadline:
            break
            
        girth, is_simple = _calc_girth_k4_lift(current, Q)
        
        if not is_simple:
            # Penalize non-simple graphs
            girth = 0 
            
        if girth >= g:
            # Found valid graph
            return _build_k4_lift(current, Q)
            
        if girth > best_girth:
            best_girth = girth
            best_voltages = current.copy()
            
        # Pick a random edge to mutate
        i = random.randint(0, 5)
        old_val = current[i]
        current[i] = random.randint(0, Q-1)
        
        new_girth, new_simple = _calc_girth_k4_lift(current, Q)
        if not new_simple:
            new_girth = 0
            
        if new_girth > girth:
            girth = new_girth
        else:
            # Revert
            current[i] = old_val

    return None

def _calc_girth_k4_lift(voltages, Q):
    # Check simplicity and girth
    # Vertices: 0..4Q-1
    # v in 0..3, g in 0..Q-1
    # id = v*Q + g
    
    # Build adjacency set to check for multi-edges
    adj_set = set()
    adj_list = [[] for _ in range(4*Q)]
    
    v01, v02, v03, v12, v13, v23 = voltages
    
    for g in range(Q):
        # 0-g
        u = 0*Q + g
        # to 1
        v = 1*Q + (g + v01) % Q
        if (u, v) in adj_set or (v, u) in adj_set:
            return 0, False
        adj_set.add((u, v))
        adj_set.add((v, u))
        adj_list[u].append(v)
        adj_list[v].append(u)
        
        # to 2
        v = 2*Q + (g + v02) % Q
        if (u, v) in adj_set or (v, u) in adj_set:
            return 0, False
        adj_set.add((u, v))
        adj_set.add((v, u))
        adj_list[u].append(v)
        adj_list[v].append(u)
        
        # to 3
        v = 3*Q + (g + v03) % Q
        if (u, v) in adj_set or (v, u) in adj_set:
            return 0, False
        adj_set.add((u, v))
        adj_set.add((v, u))
        adj_list[u].append(v)
        adj_list[v].append(u)
        
    for g in range(Q):
        # 1-g
        u = 1*Q + g
        # to 2
        v = 2*Q + (g + v12) % Q
        if (u, v) in adj_set or (v, u) in adj_set:
            return 0, False
        adj_set.add((u, v))
        adj_set.add((v, u))
        adj_list[u].append(v)
        adj_list[v].append(u)
        
        # to 3
        v = 3*Q + (g + v13) % Q
        if (u, v) in adj_set or (v, u) in adj_set:
            return 0, False
        adj_set.add((u, v))
        adj_set.add((v, u))
        adj_list[u].append(v)
        adj_list[v].append(u)
        
    for g in range(Q):
        # 2-g
        u = 2*Q + g
        # to 3
        v = 3*Q + (g + v23) % Q
        if (u, v) in adj_set or (v, u) in adj_set:
            return 0, False
        adj_set.add((u, v))
        adj_set.add((v, u))
        adj_list[u].append(v)
        adj_list[v].append(u)
        
    # Now check girth using BFS
    # We only need to check up to depth 7
    N = 4*Q
    visited = [[False]*N for _ in range(8)] # visited[depth][node]
    
    # To speed up, we can run BFS from a few vertices
    # Or just one vertex? No, girth is global.
    # But if we find a cycle, we return its length.
    # We want the MINIMUM cycle length.
    # So we check for cycles of length 2, 3, ..., 15.
    
    # Actually, BFS from each vertex up to depth 7 is O(N * 3^7) which is large.
    # Better: For each vertex, run BFS and track distances.
    # If we encounter a neighbor with dist < current_dist, we have a cycle.
    # Cycle length = dist[u] + dist[v] + 1.
    
    # We can just run BFS from vertex 0? No, the shortest cycle might not pass through 0.
    # We need to run from all vertices.
    # But 936 vertices is small.
    
    min_girth = 100
    
    for start in range(N):
        dist = [-1] * N
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= 7: continue
            for v in adj_list[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # But this might not be the shortest cycle through u,v
                    # However, BFS guarantees dist is shortest.
                    # If u and v are adjacent, and both visited, 
                    # the cycle is dist[u] + dist[v] + 1.
                    # We take the min.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth < 4: # Early exit
                        return min_girth, True
                        
    return min_girth, True

def _build_k4_lift(voltages, Q):
    # Build the edge list
    edges = []
    v01, v02, v03, v12, v13, v23 = voltages
    
    for g in range(Q):
        u = 0*Q + g
        v = 1*Q + (g + v01) % Q
        edges.append((u, v))
        v = 2*Q + (g + v02) % Q
        edges.append((u, v))
        v = 3*Q + (g + v03) % Q
        edges.append((u, v))
        
    for g in range(Q):
        u = 1*Q + g
        v = 2*Q + (g + v12) % Q
        edges.append((u, v))
        v = 3*Q + (g + v13) % Q
        edges.append((u, v))
        
    for g in range(Q):
        u = 2*Q + g
        v = 3*Q + (g + v23) % Q
        edges.append((u, v))
        
    return 4*Q, edges
