import numpy as np
import random
import time

def construct(k, g):
    if k != 4 or g != 10:
        # Fallback to seed for other parameters
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

    # Specific case k=4, g=10
    # Strategy: Tensor product C_20 x C_20 + Local Search
    
    L = 20
    n = L * L
    adj = {i: set() for i in range(n)}
    edges = []
    
    def add_edge(u, v):
        if u == v: return False
        if v in adj[u]: return False
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        return True

    def remove_edge(u, v):
        if v not in adj[u]: return False
        adj[u].remove(v)
        adj[v].remove(u)
        edges.remove((u, v))
        return True

    # Build C_L x C_L (Cartesian) or Tensor?
    # Directive says Tensor.
    # Tensor: (i,j) connected to (i+1, j+1), (i+1, j-1), (i-1, j+1), (i-1, j-1)
    for i in range(L):
        for j in range(L):
            u = i * L + j
            for di in [1, -1]:
                for dj in [1, -1]:
                    ni = (i + di) % L
                    nj = (j + dj) % L
                    v = ni * L + nj
                    if u < v:
                        add_edge(u, v)

    # Verify 4-regular
    for i in range(n):
        if len(adj[i]) != 4:
            # Fallback to seed
            return _seed_fallback(k, g)

    def get_girth():
        # BFS from each node to find shortest cycle
        # Optimization: Only need to check if girth < 10
        # Use BFS from each node, stop at depth 4 (since 2*4+1=9 < 10? No, cycle length = dist[u]+dist[v]+1. If dist <= 4, cycle <= 9)
        # Actually, if we find any cycle < 10, return it.
        for start in range(n):
            dist = {start: 0}
            parent = {start: -1}
            queue = [(start, 0)]
            while queue:
                u, d = queue.pop(0)
                if d >= 5: # Max dist for cycle < 10 is 4 (2*4+1=9)
                    break
                for v in adj[u]:
                    if v == start:
                        # Cycle length d+1
                        if d + 1 < 10:
                            return d + 1
                        continue
                    if v not in dist:
                        dist[v] = d + 1
                        parent[v] = u
                        queue.append((v, d + 1))
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < 10:
                            return cycle_len
        return 100 # >= 10

    current_girth = get_girth()
    
    start_time = time.time()
    time_limit = 100 # Leave 20s for verification/fallback
    
    while time.time() - start_time < time_limit:
        girth = get_girth()
        if girth >= 10:
            return n, edges
            
        # Find a short cycle to break
        # Re-use BFS to find one
        cycle = None
        for start in range(n):
            dist = {start: 0}
            parent = {start: -1}
            queue = [(start, 0)]
            found = False
            while queue:
                u, d = queue.pop(0)
                if d >= 5:
                    break
                for v in adj[u]:
                    if v == start:
                        if d + 1 < 10:
                            # Reconstruct cycle
                            path = []
                            curr = u
                            while curr != -1:
                                path.append(curr)
                                curr = parent[curr]
                            path.append(start)
                            cycle = path
                            found = True
                            break
                        continue
                    if v not in dist:
                        dist[v] = d + 1
                        parent[v] = u
                        queue.append((v, d + 1))
                    elif parent[u] != v and parent[v] != u:
                        cl = dist[u] + dist[v] + 1
                        if cl < 10:
                            # Reconstruct
                            path1 = []
                            curr = u
                            while curr != -1:
                                path1.append(curr)
                                curr = parent[curr]
                            path1.append(start)
                            
                            path2 = []
                            curr = v
                            while curr != -1:
                                path2.append(curr)
                                curr = parent[curr]
                            path2.append(start)
                            
                            # Combine
                            # This is complex. Just pick random edges from the graph for swapping.
                            cycle = None # Simplify: just pick random edges
                            found = True
                            break
                if found:
                    break
            if found and cycle:
                break
            if found:
                break
        
        # If no specific cycle found easily, pick random edges
        # Pick two random edges (u,v) and (x,y)
        # Try swap (u,x) and (v,y) or (u,y) and (v,x)
        
        edges_list = list(edges)
        random.shuffle(edges_list)
        swapped = False
        for _ in range(100): # Try 100 swaps per iteration
            if not edges_list: break
            e1 = edges_list.pop()
            e2 = edges_list.pop()
            u, v = e1
            x, y = e2
            # Check if u,v,x,y are distinct
            if len({u,v,x,y}) < 4: continue
            
            # Option 1: (u,x), (v,y)
            if x not in adj[u] and y not in adj[v] and x != u and y != v:
                # Remove e1, e2
                remove_edge(u, v)
                remove_edge(x, y)
                # Add new
                if add_edge(u, x) and add_edge(v, y):
                    new_girth = get_girth()
                    if new_girth >= girth:
                        # Accept
                        swapped = True
                        break
                    else:
                        # Revert
                        remove_edge(u, x)
                        remove_edge(v, y)
                        add_edge(u, v)
                        add_edge(x, y)
            
            # Option 2: (u,y), (v,x)
            if y not in adj[u] and x not in adj[v] and y != u and x != v:
                remove_edge(u, v)
                remove_edge(x, y)
                if add_edge(u, y) and add_edge(v, x):
                    new_girth = get_girth()
                    if new_girth >= girth:
                        swapped = True
                        break
                    else:
                        remove_edge(u, y)
                        remove_edge(v, x)
                        add_edge(u, v)
                        add_edge(x, y)
        
        if not swapped:
            # If no improvement in 100 tries, restart edges_list or break?
            # Continue to next outer loop iteration
            pass

    # If timeout, fallback to seed
    return _seed_fallback(k, g)

def _seed_fallback(k, g):
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
