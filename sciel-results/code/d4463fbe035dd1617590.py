import time
import sys
import random
import collections

def _try_greedy(n, k, g, deadline):
    random.seed(42)
    
    def get_dist(u, v, adj, min_dist):
        if u == v: return 0
        visited = {u: 0}
        queue = collections.deque([u])
        while queue:
            node = queue.popleft()
            dist = visited[node]
            if dist >= min_dist:
                return -1
            for nei in adj[node]:
                if nei == v:
                    return dist + 1
                if nei not in visited:
                    visited[nei] = dist + 1
                    queue.append(nei)
        return -1

    min_dist = g - 1
    max_attempts = 10
    
    for _ in range(max_attempts):
        if time.time() > deadline:
            return None
            
        deg = [0] * n
        adj = [[] for _ in range(n)]
        edges = []
        
        total_edges_needed = (n * k) // 2
        stuck_counter = 0
        
        while len(edges) < total_edges_needed:
            if time.time() > deadline:
                return None
                
            candidates = [i for i in range(n) if deg[i] < k]
            if len(candidates) < 2:
                return None
            
            u, v = random.sample(candidates, 2)
            
            if deg[u] == 0 or deg[v] == 0:
                dist = -1
            else:
                dist = get_dist(u, v, adj, min_dist)
                
            if dist == -1 or dist >= min_dist:
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
                stuck_counter = 0
            else:
                stuck_counter += 1
                if stuck_counter > 500:
                    break
                    
        if len(edges) == total_edges_needed:
            return n, edges
            
    return None

def construct(k, g):
    deadline = time.time() + 110.0
    
    # Try to find a smaller graph using greedy
    # Known best is 320
    # Try 243, 250, 300, 320
    for target_n in [243, 250, 300, 320]:
        if time.time() > deadline - 15:
            break
        res = _try_greedy(target_n, k, g, deadline - 15)
        if res is not None:
            return res
            
    # Fallback to LUW
    q = k
    if (g - 5) % 2 != 0:
        d = max(3, g - 5)
    else:
        d = max(3, g - 4)
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
