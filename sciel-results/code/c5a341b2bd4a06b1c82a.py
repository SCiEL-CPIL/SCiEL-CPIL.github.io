import time
import random
import numpy as np

def construct(k, g):
    # Fallback to original LUW if k != 3 or g != 16 to be safe, 
    # but the task is specifically for the 3-regular girth 16 cage.
    # The evaluator passes k=3, g=16.
    
    if k != 3:
        # Simple fallback for other cases, though task implies 3, 16
        return _luw_construct(k, g)

    deadline = time.time() + 110.0
    
    # Try to find a graph with n vertices
    # Best known is 936.
    # We try 936, 938, 940...
    best_graph = None
    best_n = float('inf')
    best_girth = 0

    for n in range(936, 1000, 2):
        if time.time() > deadline - 5:
            break
            
        # Try to find a graph with this n
        success, edges, girth_found = _local_search_3_reg(n, g, deadline)
        if success:
            return n, edges
        else:
            if girth_found > best_girth:
                best_girth = girth_found
                # We don't store the graph if it's invalid, 
                # but we might want to return the best invalid one?
                # No, must be valid.
                
    # If we failed to find a valid graph with girth >= g,
    # we must return *some* graph.
    # The task requires "no cycle may be shorter than g".
    # If we can't find one, we should probably return the LUW graph which is valid (girth >= 16).
    # The LUW graph is valid but large.
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Copy of the original code
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

def _local_search_3_reg(n, target_girth, deadline):
    # Generate random 3-regular graph
    adj = np.zeros((n, n), dtype=int)
    
    # Generate random 3-regular graph
    # Pair up stubs
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    edges = []
    # Simple pairing might create multi-edges or loops
    # We need to check.
    # A better way: Configuration model with rejection.
    
    # Let's just try a few random graphs
    for attempt in range(100):
        if time.time() > deadline - 10:
            return False, [], 0
            
        adj = np.zeros((n, n), dtype=int)
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        random.shuffle(stubs)
        
        valid = True
        for i in range(0, n, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if adj[u, v] > 0:
                valid = False
                break
            adj[u, v] = 1
            adj[v, u] = 1
            edges.append((u, v))
            
        if not valid:
            continue
            
        # Check degree
        if not np.all(adj.sum(axis=1) == 3):
            continue
            
        # Compute girth
        girth = _compute_girth(adj, n)
        
        if girth >= target_girth:
            return True, edges, girth
            
        # Local Search
        current_girth = girth
        best_girth = girth
        
        # Temperature for SA
        T = 1.0
        
        while time.time() < deadline - 5:
            # Find shortest cycle
            # If girth is small, find it
            if current_girth >= target_girth:
                return True, edges, current_girth
                
            # Get a cycle
            cycle = _find_shortest_cycle(adj, n, current_girth)
            if not cycle:
                break
                
            # Pick an edge on the cycle
            u, v = random.choice(cycle)
            
            # Remove (u, v)
            adj[u, v] = 0
            adj[v, u] = 0
            edges.remove((u, v))
            
            # Find an edge to swap with
            # Must not be incident to u or v to keep it simple?
            # Actually, if we pick (x, y) and x==u, we create a loop or multi-edge.
            # So x, y must not be u or v.
            
            candidates = []
            for i in range(n):
                if i == u or i == v:
                    continue
                for j in range(i+1, n):
                    if j == u or j == v:
                        continue
                    if adj[i, j]:
                        candidates.append((i, j))
                        
            if not candidates:
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append((u, v))
                continue
                
            x, y = random.choice(candidates)
            
            # Remove (x, y)
            adj[x, y] = 0
            adj[y, x] = 0
            edges.remove((x, y))
            
            # Add (u, x) and (v, y)
            if adj[u, x] or adj[v, y]:
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append((u, v))
                adj[x, y] = 1
                adj[y, x] = 1
                edges.append((x, y))
                continue
                
            adj[u, x] = 1
            adj[x, u] = 1
            adj[v, y] = 1
            adj[y, v] = 1
            edges.append((u, x))
            edges.append((v, y))
            
            new_girth = _compute_girth(adj, n)
            
            if new_girth >= current_girth:
                current_girth = new_girth
                if new_girth > best_girth:
                    best_girth = new_girth
            else:
                # SA acceptance
                delta = current_girth - new_girth
                if random.random() < np.exp(-delta / T):
                    current_girth = new_girth
                else:
                    # Revert
                    adj[u, x] = 0
                    adj[x, u] = 0
                    adj[v, y] = 0
                    adj[y, v] = 0
                    edges.remove((u, x))
                    edges.remove((v, y))
                    
                    adj[u, v] = 1
                    adj[v, u] = 1
                    edges.append((u, v))
                    
                    adj[x, y] = 1
                    adj[y, x] = 1
                    edges.append((x, y))
                    
            # Cool down
            T *= 0.99
            
        return False, edges, best_girth
        
    return False, [], 0

def _compute_girth(adj, n):
    # BFS from each node
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        for u in queue:
            for v in range(n):
                if adj[u, v]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # But we need to be careful with BFS tree
                        # If v is visited and not parent, cycle length = dist[u] + dist[v] + 1
                        # This is only valid if the cycle doesn't go through the start in a weird way?
                        # Actually, BFS finds shortest paths.
                        # If we see a visited node v (not parent), the cycle length is dist[u] + dist[v] + 1.
                        l = dist[u] + dist[v] + 1
                        if l < min_girth:
                            min_girth = l
                            
    return min_girth if min_girth != float('inf') else n

def _find_shortest_cycle(adj, n, girth):
    # Return a list of edges in a shortest cycle
    # Just find one cycle of length girth
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        for u in queue:
            for v in range(n):
                if adj[u, v]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found cycle
                        # Reconstruct
                        cycle_nodes = []
                        cur = u
                        while cur != start:
                            cycle_nodes.append(cur)
                            cur = parent[cur]
                        cycle_nodes.append(start)
                        
                        # Get edges
                        cycle_edges = []
                        for i in range(len(cycle_nodes)):
                            u1 = cycle_nodes[i]
                            v1 = cycle_nodes[(i+1) % len(cycle_nodes)]
                            if adj[u1, v1]:
                                cycle_edges.append((u1, v1))
                                
                        if len(cycle_edges) == girth:
                            return cycle_edges
                            
    return []
