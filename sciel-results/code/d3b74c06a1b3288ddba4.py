import time
import random
import sys
from collections import deque

def construct(k, g):
    # Fallback: LUW construction
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

    def neighbours_luw(vertex):
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
        for other in neighbours_luw(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    n_luw, edges_luw = len(order), edges
    
    # Attempt Greedy Random Search
    deadline = time.time() + 110
    best_n = n_luw
    best_edges = edges_luw
    
    # Greedy: grow a k-regular graph by adding vertices
    def try_greedy(max_attempts=20):
        local_best_n = best_n
        local_best_edges = best_edges
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Start with a single vertex
            adj = {0: set()}
            n = 1
            free = {0: k}
            active = [0]
            
            success = True
            max_steps = 3000
            steps = 0
            
            while free and steps < max_steps:
                steps += 1
                if time.time() > deadline:
                    success = False
                    break
                
                # Pick an active vertex with the most free slots
                active.sort(key=lambda x: -free[x])
                u = active[0]
                
                # Try to add a new vertex connected to u
                new_v = n
                n += 1
                adj[new_v] = {u}
                adj[u].add(new_v)
                free[new_v] = k - 1
                free[u] -= 1
                
                if free[u] == 0:
                    active.remove(u)
                active.append(new_v)
                
                # Now try to fill new_v's remaining slots
                for _ in range(k - 1):
                    if free[new_v] == 0:
                        break
                    
                    # Pick a random active vertex (not new_v)
                    candidates = [v for v in active if v != new_v and free[v] > 0]
                    if not candidates:
                        success = False
                        break
                    
                    # Try multiple candidates
                    random.shuffle(candidates)
                    connected = False
                    for v in candidates:
                        # Check if already connected
                        if v in adj[new_v]:
                            continue
                        
                        # Check cycle length: BFS from new_v to v
                        # If dist < g - 1, adding edge creates cycle < g
                        dist = -1
                        visited = {new_v: 0}
                        queue = deque([new_v])
                        found = False
                        while queue and not found:
                            curr = queue.popleft()
                            if visited[curr] >= g - 1:
                                continue
                            for nb in adj[curr]:
                                if nb not in visited:
                                    visited[nb] = visited[curr] + 1
                                    if nb == v:
                                        dist = visited[nb]
                                        found = True
                                        break
                                    queue.append(nb)
                        
                        if not found:
                            continue
                        
                        if dist < g - 1:
                            continue
                        
                        # Valid edge
                        adj[new_v].add(v)
                        adj[v].add(new_v)
                        free[new_v] -= 1
                        free[v] -= 1
                        
                        if free[v] == 0 and v in active:
                            active.remove(v)
                        
                        connected = True
                        break
                    
                    if not connected:
                        success = False
                        break
                
                if not success:
                    break
            
            if success and all(free[v] == 0 for v in range(n)) and n < local_best_n:
                # Build edge list
                edge_set = set()
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_set.add((u, v))
                local_best_n = n
                local_best_edges = list(edge_set)
        
        return local_best_n, local_best_edges
    
    best_n, best_edges = try_greedy()
    
    return best_n, best_edges
