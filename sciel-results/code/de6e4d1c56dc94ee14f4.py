import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    # Fallback: LUW
    def luw_construct(k, g):
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

    best_result = luw_construct(k, g)
    best_n = best_result[0]
    
    # Try to find a smaller graph
    # Target sizes: 1000, 1100, 1200...
    # We'll try a few sizes.
    
    def generate_random_regular_graph(n, k):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = []
        adj = [[] for _ in range(n)]
        valid = True
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
        if not valid:
            return None, None
        return n, edges, adj

    def check_girth(n, adj, g):
        max_depth = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def get_girth(n, adj):
        # Returns the actual girth
        max_depth = n // 2
        min_girth = max_depth + 1
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    # Switching Search
    # Try a few N values
    for target_n in [1000, 1100, 1200, 1500]:
        if target_n >= best_n:
            continue
        if time.time() > deadline - 20:
            break
            
        # Generate a random graph
        res = generate_random_regular_graph(target_n, k)
        if res is None:
            continue
        n, edges, adj = res
        
        # Check initial girth
        current_girth = get_girth(n, adj)
        
        if current_girth >= g:
            return n, edges
            
        # Local Search
        # We try to increase girth
        # Time limit for search: 10 seconds per target_n
        search_deadline = time.time() + 10
        
        while time.time() < search_deadline:
            # Pick random edges to switch
            # Pick two random edges (u, v) and (x, y)
            # Ensure they are not adjacent
            # Swap to (u, x) and (v, y)
            
            # Randomly pick 2 edges
            e1_idx = random.randint(0, len(edges) - 1)
            e2_idx = random.randint(0, len(edges) - 1)
            if e1_idx == e2_idx:
                continue
                
            u, v = edges[e1_idx]
            x, y = edges[e2_idx]
            
            # Check if edges are adjacent (share a vertex)
            if u == x or u == y or v == x or v == y:
                continue
                
            # Check if new edges already exist
            # This is O(k) if we use sets, but adj is list.
            # For simplicity, we assume they don't exist or check quickly.
            # If they exist, switch creates multi-edge.
            # We can check `x in adj[u]` etc.
            if x in adj[u] or y in adj[v] or x in adj[v] or y in adj[u]:
                continue
                
            # Perform switch
            # Remove old edges
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            # Add new edges
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            # Update edges list
            edges[e1_idx] = (u, x)
            edges[e2_idx] = (v, y)
            
            # Check new girth
            new_girth = get_girth(n, adj)
            
            if new_girth > current_girth:
                current_girth = new_girth
                if current_girth >= g:
                    return n, edges
            else:
                # Revert
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
                
                edges[e1_idx] = (u, v)
                edges[e2_idx] = (x, y)
                
                # Accept with small probability to escape local minima?
                # For now, strict hill climbing.

    return best_result
