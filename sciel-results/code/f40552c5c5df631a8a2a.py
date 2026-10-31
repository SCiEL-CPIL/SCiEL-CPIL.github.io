import time
import random
import sys

def seed_construct(k, g):
    # Copy of the seed's logic from the prompt
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

def generate_cubic(n):
    if n < 4 or n % 2 != 0:
        return None
    
    # Start with a cycle
    adj = [[(i-1)%n, (i+1)%n] for i in range(n)]
    
    # We need to add n/2 edges.
    # Each vertex has 1 slot left.
    # Slots are indexed by vertex id.
    # We need a perfect matching on n vertices, avoiding existing edges.
    
    for _ in range(100): # Retry
        slots = list(range(n))
        random.shuffle(slots)
        adj_temp = [list(neighs) for neighs in adj]
        success = True
        for i in range(0, n, 2):
            u = slots[i]
            v = slots[i+1]
            if u == v:
                success = False
                break
            if v in adj_temp[u]:
                success = False
                break
            adj_temp[u].append(v)
            adj_temp[v].append(u)
        if success:
            return adj_temp
    return None

def check_girth(adj, n, g):
    """
    Returns the girth if < g, or g if girth >= g.
    """
    for start in range(n):
        dist = [-1] * n
        prev = [-1] * n
        queue = [start]
        dist[start] = 0
        while queue:
            u = queue.pop(0)
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    queue.append(v)
                elif v != prev[u] and v != start:
                    # Found a cycle
                    length = dist[u] + dist[v] + 1
                    if length < g:
                        return length
                elif v == start:
                    length = dist[u] + 1
                    if length < g:
                        return length
    return g

def find_short_cycle(adj, n, g):
    """
    Returns a list of vertices in a short cycle, or None.
    """
    for start in range(n):
        dist = [-1] * n
        prev = [-1] * n
        queue = [start]
        dist[start] = 0
        while queue:
            u = queue.pop(0)
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    queue.append(v)
                elif v != prev[u] and v != start:
                    length = dist[u] + dist[v] + 1
                    if length < g:
                        # Reconstruct cycle
                        path1 = []
                        curr = u
                        while curr != -1:
                            path1.append(curr)
                            if curr == start: break
                            curr = prev[curr]
                        path1.reverse()
                        
                        path2 = []
                        curr = v
                        while curr != -1:
                            path2.append(curr)
                            if curr == start: break
                            curr = prev[curr]
                        
                        # Cycle is path1 + path2[1:]
                        # But we need to be careful with order.
                        # path1: start -> ... -> u
                        # path2: start -> ... -> v
                        # Cycle: u -> ... -> start -> ... -> v -> u
                        # Vertices: path1 + path2[1:]
                        cycle = path1 + path2[1:]
                        # Remove duplicates? 
                        # path1 ends with u. path2 starts with v? No, path2 is from v to start?
                        # My BFS stores prev[v] = u, so prev points towards start.
                        # So path1 is start -> ... -> u.
                        # path2 is start -> ... -> v.
                        # The cycle is u -> ... -> start -> ... -> v -> u.
                        # The list of vertices in the cycle is unique.
                        return cycle
                elif v == start:
                    length = dist[u] + 1
                    if length < g:
                        path = []
                        curr = u
                        while curr != -1:
                            path.append(curr)
                            if curr == start: break
                            curr = prev[curr]
                        path.reverse()
                        return path
    return None

def apply_switch(adj, u, v, x, y):
    # Check validity
    if u == x or u == y or v == x or v == y:
        return False
    if x in adj[u] or y in adj[v]:
        # Option 1 invalid
        pass
    else:
        # Try Option 1: (u,x), (v,y)
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(y)
        adj[y].append(v)
        return True

    # Revert if we entered the block? No, we only modify if we pass the check.
    # Wait, my logic above was flawed. I need to check before modifying.
    
    # Let's restart apply_switch logic cleanly.
    # Check Option 1: (u,x) and (v,y)
    if x not in adj[u] and y not in adj[v] and u != x and v != y:
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(y)
        adj[y].append(v)
        return True
        
    # Check Option 2: (u,y) and (v,x)
    if y not in adj[u] and x not in adj[v] and u != y and v != x:
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].append(y)
        adj[y].append(u)
        adj[v].append(x)
        adj[x].append(v)
        return True
        
    return False

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find a small graph
    for n in range(512, 2000, 2):
        if time.time() > deadline - 5:
            break
            
        n_start = time.time()
        
        adj = generate_cubic(n)
        if not adj:
            continue
            
        # Local Search
        while time.time() - n_start < 5.0:
            if time.time() > deadline:
                break
                
            min_girth = check_girth(adj, n, g)
            if min_girth >= g:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            else:
                # Try to improve
                cycle = find_short_cycle(adj, n, g)
                if not cycle:
                    break
                    
                # Pick a random edge in the cycle
                # Cycle is a list of vertices.
                # Edges are (cycle[i], cycle[i+1]) and (cycle[-1], cycle[0])
                # But my reconstruction might not be perfectly cyclic order?
                # path1 + path2[1:] is a path from u to v via start.
                # It is not a closed cycle list.
                # I need to get the actual edges.
                
                # Let's just pick a random vertex u in the cycle, and a neighbor v in the cycle.
                # Since it's a cycle, every vertex has 2 neighbors in the cycle.
                # I need to identify which neighbors are in the cycle.
                
                # Simpler: Pick a random edge (u, v) from the entire graph.
                # If it's in a short cycle, switching it might help.
                # But we need to know if it's in a short cycle.
                
                # Alternative: Just pick a random edge (u, v) and a random edge (x, y).
                # If the switch is valid, accept it.
                # This is a random walk.
                
                all_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            all_edges.append((i, j))
                
                if not all_edges:
                    break
                    
                u, v = random.choice(all_edges)
                x, y = random.choice(all_edges)
                
                if (u, v) == (x, y) or (u, v) == (y, x):
                    continue
                    
                if apply_switch(adj, u, v, x, y):
                    pass
                else:
                    pass
                    
    # Fallback
    return seed_construct(k, g)
