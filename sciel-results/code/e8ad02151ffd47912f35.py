import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # --- L-U-W Construction Logic ---
    
    def get_luw_params(q, d):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        return add, sub, mul, rule

    def build_luw(k, d):
        q = k
        add, sub, mul, rule = get_luw_params(q, d)
        
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

    def verify_girth(n, edges, target_g):
        if n < target_g:
            return False
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles of length < target_g
        # BFS from each vertex
        for u in range(n):
            if time.time() > deadline:
                return None  # Timeout signal
            
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                x = q.popleft()
                # If we are already at half the target girth, 
                # any cycle through x would be at least 2*dist[x] + 1?
                # Actually, if dist[x] >= target_g // 2, 
                # a back edge to y with dist[y] >= 0 gives cycle >= 2*dist[x] + 1?
                # Not necessarily. But we can stop expanding.
                if dist[x] >= target_g // 2:
                    continue
                    
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        q.append(y)
                    elif y != parent[x]:
                        # Found a back edge
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < target_g:
                            return False
        return True

    # --- Search Strategy ---
    
    # Calculate seed's d for fallback
    d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Try smaller d values first
    # L-U-W girth is roughly 2d+2. 
    # So for g, we need 2d+2 >= g => d >= (g-2)/2.
    # Let's start from max(3, (g-2)//2 - 2) to be safe.
    min_d = max(3, (g - 2) // 2 - 2)
    
    best_n = None
    best_edges = None
    
    for d in range(min_d, d_seed + 1):
        if time.time() > deadline:
            break
            
        n, edges = build_luw(k, d)
        
        # Quick check: if n is already huge, maybe skip verification if we have a candidate?
        # But we want the smallest n.
        
        result = verify_girth(n, edges, g)
        
        if result is True:
            best_n = n
            best_edges = edges
            break # Found valid graph with this d, which is smaller than next d
            
    if best_n is not None:
        return best_n, best_edges
        
    # Fallback to seed's construction
    n, edges = build_luw(k, d_seed)
    return n, edges
