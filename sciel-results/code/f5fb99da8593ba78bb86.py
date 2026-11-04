import time
import sys

def _girth_of_gpg(n, k):
    """
    Calculate the girth of the Generalized Petersen Graph G(n, k).
    Vertices: u_0..u_{n-1}, v_0..v_{n-1}
    Edges: (u_i, u_{i+1}), (u_i, v_i), (v_i, v_{i+k})
    """
    if n < 5:
        return 0
    
    # Build adjacency list
    adj = [[] for _ in range(2 * n)]
    for i in range(n):
        adj[i].append((i + 1) % n)
        adj[i].append(i + n)
        adj[i + n].append(i)
        adj[i + n].append((i + k) % n + n)
    
    # Ensure no self-loops or multi-edges in construction (GPGs are simple for n>=5, 1<=k<=n/2)
    # But we should check for simplicity just in case
    # For simplicity, we assume standard GPG conditions hold.
    
    # Check girth using BFS from each vertex
    min_girth = float('inf')
    
    # We can prune: if we find a cycle of length L, min_girth = L.
    # We stop if min_girth < 16.
    
    for start in range(2 * n):
        # BFS
        visited = {start: (None, 0)} # node: (parent, depth)
        queue = [(start, None, 0)]
        found_cycle = False
        
        while queue:
            node, parent, depth = queue.pop(0)
            if depth >= 8: # Max depth needed for girth 16 is 8 (cycle length 2*depth)
                continue
            
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    p_node, p_depth = visited[neighbor]
                    # If we found a cycle
                    cycle_len = depth + p_depth + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth < 16:
                        return min_girth
                else:
                    visited[neighbor] = (node, depth + 1)
                    queue.append((neighbor, node, depth + 1))
                    
    return min_girth

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try Generalized Petersen Graphs
    # Moore bound for k=3, g=16 is 510. So 2n >= 510 => n >= 255.
    # Known best is 936 => n=468.
    # Search n from 255 to 468.
    for n in range(255, 469):
        if time.time() > deadline:
            break
        for k_val in range(1, n // 2 + 1):
            if time.time() > deadline:
                break
            
            # Quick check: cycle length of v-ring is n/gcd(n,k)
            import math
            g_v = n // math.gcd(n, k_val)
            if g_v < g:
                continue
            
            # Quick check: cycle length of u-ring is n
            if n < g:
                continue
            
            # Check girth
            girth = _girth_of_gpg(n, k_val)
            if girth >= g:
                # Construct edges
                edges = []
                for i in range(n):
                    edges.append((i, (i + 1) % n))
                    edges.append((i, i + n))
                    edges.append((i + n, (i + k_val) % n + n))
                return 2 * n, edges
    
    # Fallback to LUW construction if no GPG found
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
