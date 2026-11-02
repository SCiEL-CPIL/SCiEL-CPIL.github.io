import time
import random
import collections
import sys

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct():
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

    # Get baseline
    n_luw, edges_luw = luw_construct()
    best_n = n_luw
    best_edges = edges_luw

    # Random search for smaller graphs
    # Moore bound approximation
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = k + (k - 1) * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r))
        
    min_n = max(moore + 1, 400)
    
    # Try sizes from min_n up to best_n - 1
    # We only need to try if we have time
    for n in range(min_n, best_n):
        if time.time() > deadline - 5.0:
            break
            
        # Try a few random graphs for this size
        attempts = 0
        max_attempts = 50 if n < 1000 else 10
        
        while attempts < max_attempts and time.time() < deadline - 5.0:
            attempts += 1
            edges, adj = generate_random_regular_graph(n, k)
            if edges is None:
                continue
            
            # Check girth
            if not has_short_cycle(adj, g):
                # Check connectivity (BFS from 0)
                if is_connected(adj, n):
                    best_n = n
                    best_edges = edges
                    # We found a smaller one, can we find even smaller?
                    # The loop increments n, so we are looking for the *smallest* n.
                    # If we found one at n, we should check if there's one at n-1?
                    # The loop goes upwards. So if we find one at n=500, we return it?
                    # No, we want the *minimum*.
                    # If we iterate n from low to high, the first valid one is the minimum.
                    return best_n, best_edges
            
    return best_n, best_edges

def generate_random_regular_graph(n, k):
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    edges = set()
    adj = [[] for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        
        if u == v:
            return None, None # Loop
        
        if u > v:
            u, v = v, u
            
        if (u, v) in edges:
            return None, None # Multi-edge
            
        edges.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
        
    return list(edges), adj

def has_short_cycle(adj, g):
    n = len(adj)
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = collections.deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                else:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    # This is the length of the cycle formed by the two tree paths and the edge (u,v)
                    # If this length is < g, we have a short cycle
                    # Note: This might detect a cycle that is not the shortest, but if ANY cycle is short, 
                    # this condition will trigger for some s, u, v.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return True
    return False

def is_connected(adj, n):
    if n == 0:
        return True
    visited = [False] * n
    q = collections.deque([0])
    visited[0] = True
    count = 1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                q.append(v)
    return count == n

# EVOLVE-BLOCK-END
