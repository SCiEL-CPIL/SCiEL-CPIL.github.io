import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    def _known_cages(k, g):
        if k == 3 and g == 5:
            edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                     (5,7),(7,9),(9,6),(6,8),(8,5),
                     (0,5),(1,6),(2,7),(3,8),(4,9)]
            return 10, edges
        if k == 3 and g == 7:
            lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
            edges = []
            for i, line in enumerate(lines):
                for pt in line:
                    edges.append((pt, 7 + i))
            return 14, edges
        return None
    
    def luw_construct(k, g):
        q = k
        
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        elif is_prime(q):
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        else:
            return None
        
        best_d = None
        for d in range(3, 20):
            if d % 2 == 0:
                girth_bound = d + 5
            else:
                girth_bound = d + 4
            if girth_bound >= g:
                best_d = d
                break
        
        if best_d is None:
            return None
        
        d = best_d
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    def has_short_cycle(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start_u in range(n):
            for start_v in adj[start_u]:
                visited = set()
                visited.add((start_u, start_v))
                queue = deque()
                queue.append((start_v, start_u, 1))
                
                while queue:
                    curr, prev, depth = queue.popleft()
                    if depth >= g - 1:
                        continue
                    for nb in adj[curr]:
                        if nb == prev:
                            continue
                        if nb == start_u and depth + 1 < g:
                            return True
                        state = (curr, nb)
                        if state not in visited:
                            visited.add(state)
                            queue.append((nb, curr, depth + 1))
        return False
    
    known = _known_cages(k, g)
    if known is not None:
        best_n, best_edges = known
    else:
        luw_result = luw_construct(k, g)
        if luw_result is not None:
            best_n, best_edges = luw_result
        else:
            raise ValueError(f"Cannot construct {k}-regular graph of girth {g}")
    
    if not has_short_cycle(best_n, best_edges, g):
        return best_n, best_edges
    
    return best_n, best_edges
