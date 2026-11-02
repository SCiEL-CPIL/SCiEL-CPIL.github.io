import time
import random
import numpy as np

_DEADLINE = 110.0

def construct(k, g):
    _t0 = time.time()
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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
            if time.time() - _t0 > _DEADLINE:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # If the task is not the specific (4,10) cage, use LUW
    if k != 4 or g != 10:
        return luw_construct(k, g)

    # Specialized construction for (4,10)-cage
    # The (4,10)-cage is the point-line incidence graph of the generalized quadrangle GQ(2,2)
    # Actually, GQ(2,2) has 15 points and 15 lines, giving a 30-vertex graph (the Doetsch-Higman graph)
    # which is 4-regular with girth 8.
    
    # For girth 10, we need a different construction.
    # The smallest known (4,10)-cage has 320 vertices.
    # One construction uses a voltage lift of the Petersen graph.
    
    # Let's try a deterministic construction based on algebraic methods.
    # The (4,10)-cage can be constructed as a Cayley graph or using specific algebraic structures.
    
    # Alternative approach: Use the known 320-vertex construction.
    # The 320-vertex (4,10)-cage was found by Exoo, Goedgebeur, Jooken, Stubbe and Van den Eede.
    # It can be described using a specific set of generators in a group.
    
    # For now, let's try a random search approach with better heuristics.
    n_target = 320
    
    def random_regular_graph(n, k):
        # Configuration model with rejection
        for _ in range(100):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            valid = True
            seen = set()
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append((u, v))
            if valid:
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                return adj, edges
        return None, None

    def has_girth_at_least(n, adj, g):
        # Use BFS from each vertex to check girth
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    # Try a few random starts with local search
    deadline = _t0 + _DEADLINE
    
    # First, try a structured construction
    # The (4,10)-cage can be constructed using a specific algebraic method.
    # Let's try the construction based on the group Z_5 x Z_5 x Z_5 x Z_5 or similar.
    
    # Actually, let's use a known construction: the (4,10)-cage is a Cayley graph of a group of order 320.
    # One such group is Z_5 x Z_5 x Z_5 x Z_5 with specific generators.
    
    # For now, let's try random search with a time limit.
    best_adj = None
    best_edges = None
    
    for start_attempt in range(50):
        if time.time() > deadline:
            break
            
        adj, edges = random_regular_graph(n_target, k)
        if adj is None:
            continue
        
        # Check girth
        if has_girth_at_least(n_target, adj, g):
            return n_target, edges
        
        # Local Search: try to break short cycles
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Find a short cycle using BFS from a random vertex
            start = random.randint(0, n_target - 1)
            dist = [-1] * n_target
            parent = [-1] * n_target
            dist[start] = 0
            queue = [start]
            head = 0
            short_cycle_found = False
            cycle_edge = None
            
            while head < len(queue) and not short_cycle_found:
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < g:
                            short_cycle_found = True
                            cycle_edge = (u, v)
                            break
            
            if not short_cycle_found:
                # No short cycle found from this start, try another start
                continue
            
            # Try to break the cycle by swapping edges
            u, v = cycle_edge
            swapped = False
            
            for _ in range(50):
                # Pick random edge (w, z)
                w = random.randint(0, n_target - 1)
                if not adj[w]:
                    continue
                z = random.choice(adj[w])
                
                # Ensure (w, z) is not the same as (u, v)
                if (w == u and z == v) or (w == v and z == u):
                    continue
                
                # Option 1: swap (u, v) and (w, z) to (u, w) and (v, z)
                if u != w and v != z:
                    if w not in adj[u] and z not in adj[v]:
                        # Perform swap
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[w].remove(z)
                        adj[z].remove(w)
                        
                        adj[u].append(w)
                        adj[w].append(u)
                        adj[v].append(z)
                        adj[z].append(v)
                        
                        swapped = True
                        break
                
                # Option 2: swap (u, v) and (w, z) to (u, z) and (v, w)
                if u != z and v != w:
                    if z not in adj[u] and w not in adj[v]:
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[w].remove(z)
                        adj[z].remove(w)
                        
                        adj[u].append(z)
                        adj[z].append(u)
                        adj[v].append(w)
                        adj[w].append(v)
                        
                        swapped = True
                        break
            
            if not swapped:
                continue
        
        # After local search, check again
        if has_girth_at_least(n_target, adj, g):
            edges = []
            for u in range(n_target):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n_target, edges

    # Fallback
    return luw_construct(k, g)
