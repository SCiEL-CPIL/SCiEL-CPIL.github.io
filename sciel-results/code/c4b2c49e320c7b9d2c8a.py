import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Construct LUW graph as baseline
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        mult_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        def add(x, y): return add_table[x][y]
        def sub(x, y): return add_table[x][y]
        def mul(x, y): return mult_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    rule = [None] * (d + 1)
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0: rule[c] = (1, c0 - 2)
        elif r == 1: rule[c] = (c0 - 1, 1)
        elif r == 2: rule[c] = (c0, 1)
        else: rule[c] = (1, c0 + 1)
    
    def get_neighbours(side, x):
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = 0
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    luw_edges = []
    
    for vertex in order:
        if time.time() > deadline: break
        side, x = vertex
        for other in get_neighbours(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                luw_edges.append((index[vertex], index[other]))
                
    luw_n = len(order)
    
    # 2. Try to find a smaller graph via random search with verified girth
    best_n = luw_n
    best_edges = luw_edges
    
    def check_girth_fast(adj, n, g):
        """Check if graph has girth >= g using non-backtracking BFS.
        Returns True if girth >= g, False otherwise."""
        # For each start vertex s, do BFS on non-backtracking states
        # State: (u, prev) where prev is the vertex we came from
        # Start: (s, t) for each neighbor t of s, dist = 1
        # If we reach (s, u) where u is a neighbor of s and u != the original start prev,
        # and dist < g, we have a cycle.
        
        # More precisely: a cycle of length L < g means there exists s and a 
        # non-backtracking walk of length L from s back to s.
        # In terms of states: start at (s, t1), end at (s, tL) where tL != t1
        # (if tL == t1, it's just backtracking).
        # Actually, we need to be careful. Let's use the standard approach:
        # For each s, BFS from all (s, t) for t in adj[s].
        # If during BFS we reach state (s, u) where u != the prev that started the path
        # to (s, ...), and the distance is < g, we have a cycle.
        
        # Simpler: just check if any non-backtracking walk of length < g returns to start.
        # For each s:
        #   visited = set()  # (u, prev) pairs
        #   queue = deque()
        #   for t in adj[s]:
        #       queue.append((s, t, 1))  # at s, came from t, dist 1
        #       visited.add((s, t))
        #   while queue:
        #       u, prev, dist = queue.popleft()
        #       if dist >= g: continue
        #       for v in adj[u]:
        #           if v == prev: continue  # non-backtracking
        #           if v == s and dist + 1 >= 3:
        #               # We found a cycle of length dist + 1
        #               if dist + 1 < g:
        #                   return False
        #           if (v, u) not in visited:
        #               visited.add((v, u))
        #               queue.append((v, u, dist + 1))
        #   # No cycle found from s
        # return True
        
        # This is O(n * k * (k-1)^(g/2)) which for n=1000, k=5, g=9 is about 1.28e6
        # But in Python, this might be slow. Let's optimize.
        
        for s in range(n):
            visited = set()
            queue = deque()
            for t in adj[s]:
                queue.append((s, t, 1))
                visited.add((s, t))
            while queue:
                u, prev, dist = queue.popleft()
                if dist >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s and dist + 1 >= 3:
                        if dist + 1 < g:
                            return False
                    if (v, u) not in visited:
                        visited.add((v, u))
                        queue.append((v, u, dist + 1))
        return True
    
    # Build adjacency from LUW edges for reference
    luw_adj = [[] for _ in range(luw_n)]
    for u, v in luw_edges:
        luw_adj[u].append(v)
        luw_adj[v].append(u)
    
    # Try random regular graphs with n < luw_n
    # We'll try a few sizes: luw_n - 1, luw_n - 2, ..., down to luw_n - 10
    # For each size, try a few random constructions
    
    def random_regular_graph(n, k, max_trials=10):
        """Try to construct a random k-regular graph on n vertices."""
        if n * k % 2 != 0:
            return None
        if n < k + 1:
            return None
        
        for trial in range(max_trials):
            # Configuration model
            half_edges = list(range(n * k))
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            edges_set = set()
            valid = True
            
            for i in range(0, n * k, 2):
                u = half_edges[i] // k
                v = half_edges[i + 1] // k
                if u == v:
                    valid = False
                    break
                if (u, v) in edges_set or (v, u) in edges_set:
                    valid = False
                    break
                edges_set.add((u, v))
                adj[u].append(v)
                adj[v].append(u)
            
            if not valid:
                continue
            
            # Check degree
            if any(len(adj[i]) != k for i in range(n)):
                continue
            
            # Check girth
            if check_girth_fast(adj, n, g):
                # Convert to edge list
                edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append((u, v))
                return n, edge_list
        
        return None
    
    # Try smaller graphs
    for target_n in range(luw_n - 1, max(luw_n - 15, 0), -1):
        if time.time() > deadline - 10:
            break
        result = random_regular_graph(target_n, k, max_trials=5)
        if result is not None:
            best_n, best_edges = result
            break
    
    return best_n, best_edges
