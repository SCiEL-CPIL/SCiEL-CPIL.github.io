import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Use LUW construction with optimal d
    # For girth g, we need 2d + 1 >= g => d >= (g-1)/2
    d = (g - 1) // 2
    if d < 2:
        d = 2
        
    return _luw_construct(k, g, d, deadline)

def _luw_construct(k, g, d, deadline):
    q = k
    
    # Define operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
    # Define rules for the LUW recurrence
    # y_c = x_c + y_a * x_b (for side 0)
    # y_c = x_c - x_a * y_b (for side 1)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        # Standard LUW rule pattern
        # The pattern in the seed program was:
        # r = (c - 5) % 4
        # c0 = c - (c - 5) % 4
        # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        # This ensures indices are valid (1..c-1)
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # Ensure indices are within 1..c-1
        # The formula in seed:
        # r=0: (1, c0-2). If c=5, c0=5, (1, 3). OK.
        # r=1: (c0-1, 1). If c=6, c0=5, (4, 1). OK.
        # r=2: (c0, 1). If c=7, c0=5, (5, 1). OK.
        # r=3: (1, c0+1). If c=8, c0=5, (1, 6). OK.
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            # x is a tuple of length d+1: (None, x1, ..., xd)
            # y is a list of length d+1: (None, y1, ..., yd)
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
        
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            break
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
        
    n = len(order)
    # Filter edges to ensure they are within bounds (though they should be)
    valid_edges = [(u, v) for u, v in edges if u < n and v < n]
    
    return n, valid_edges

def _check_girth(edges, n, min_girth):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True
