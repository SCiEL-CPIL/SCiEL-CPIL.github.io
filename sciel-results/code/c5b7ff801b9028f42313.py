import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: LUW graph
    def get_luw():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # Try to find a better graph
    # We try a few sizes
    for n_target in [936, 1000, 1200, 1500]:
        if time.time() > deadline:
            break
        
        # Generate random 3-regular graph
        adj = generate_random_3_reg(n_target)
        if adj is None:
            continue
            
        current_girth = compute_girth(adj, n_target)
        best_adj = adj
        best_girth = current_girth
        
        if current_girth >= g:
            return n_target, adj_to_edges(adj)
            
        # Hill climbing
        while time.time() < deadline - 5:
            # Try a random 2-switch
            new_adj = try_2_switch(adj, n_target)
            if new_adj is None:
                continue
            
            new_girth = compute_girth(new_adj, n_target)
            
            if new_girth > best_girth:
                best_girth = new_girth
                best_adj = new_adj
                adj = new_adj
                if new_girth >= g:
                    return n_target, adj_to_edges(adj)
            elif new_girth == best_girth:
                # Accept with some probability to escape local minima
                if random.random() < 0.1:
                    adj = new_adj
                    best_adj = new_adj
            # If new_girth < best_girth, reject (keep adj as is, but maybe we should revert adj to best_adj?)
            # Actually, if we rejected, adj is still the old one.
            # But we might want to explore from best_adj.
            # Let's always explore from the current 'adj' which is either best or a random step.
            # To be safe, if we reject, we stay at the previous state.
            
        # If we found a graph with girth close to g, but not quite, we can't return it.
        # We must return a valid graph.
        
    # Fallback
    return get_luw()

def generate_random_3_reg(n):
    # Random pairing of half-edges
    half_edges = []
    for i in range(n):
        half_edges.extend([i, i, i])
    
    # Shuffle
    random.shuffle(half_edges)
    
    adj = [set() for _ in range(n)]
    
    for i in range(0, 3*n, 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            return None # Loop
        if v in adj[u]:
            return None # Multi-edge
        adj[u].add(v)
        adj[v].add(u)
        
    return adj

def compute_girth(adj, n):
    # BFS from each vertex
    # Returns the shortest cycle length
    min_girth = n + 1
    
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
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # But this is only valid if u and v are in the same BFS tree
                    # And we need to ensure it's a simple cycle
                    # Actually, BFS finds the shortest path from start.
                    # If we find an edge (u,v) where dist[v] <= dist[u] (or vice versa),
                    # the cycle length is dist[u] + dist[v] + 1.
                    # We take the minimum over all such edges.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        
    return min_girth

def adj_to_edges(adj):
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def try_2_switch(adj, n):
    # Pick a random edge (u, v)
    # Pick another random edge (x, y)
    # Remove them, add (u, x) and (v, y)
    # Ensure no loops or multi-edges
    
    # Flatten edges for random selection
    # This is inefficient, but for n=1000, it's okay.
    # Better: pick u, then pick v from adj[u].
    u = random.randint(0, n-1)
    if not adj[u]:
        return None
    v = random.choice(list(adj[u]))
    
    # Pick another edge (x, y) not involving u or v
    # Just pick random x, y
    x = random.randint(0, n-1)
    if x == u or x == v:
        return None
    if not adj[x]:
        return None
    y = random.choice(list(adj[x]))
    if y == u or y == v:
        return None
    if x == y:
        return None
        
    # Check if (u,v) and (x,y) are the same edge
    if (u == x and v == y) or (u == y and v == x):
        return None
        
    # Perform switch
    new_adj = [set(s) for s in adj]
    
    # Remove
    new_adj[u].remove(v)
    new_adj[v].remove(u)
    new_adj[x].remove(y)
    new_adj[y].remove(x)
    
    # Add
    if x in new_adj[u] or y in new_adj[v]:
        return None # Multi-edge
    if u == x or v == y:
        return None # Loop
        
    new_adj[u].add(x)
    new_adj[x].add(u)
    new_adj[v].add(y)
    new_adj[y].add(v)
    
    return new_adj
