import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0
    
    if k != 3:
        return 0, []
    
    q = 3
    
    def _check_girth_fast(n, edges, g):
        """Check if girth >= g using non-backtracking BFS."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each directed edge (u,v), do BFS in state space (curr, prev)
        # Check if we can return to (u,v) in fewer than g steps
        for u in range(n):
            for v in adj[u]:
                # BFS from (v, u), target is (u, v)
                # Cycle length = path_length + 1
                # Need cycle length >= g, so path_length >= g-1
                visited = set()
                queue = deque()
                visited.add((v, u))
                queue.append((v, u, 0))
                
                found_short = False
                while queue and not found_short:
                    curr, prev, depth = queue.popleft()
                    if depth >= g - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        ns = (nxt, curr)
                        if ns == (u, v):
                            if depth + 1 < g:
                                return False
                            else:
                                found_short = True
                                break
                        if ns not in visited:
                            visited.add(ns)
                            queue.append((nxt, curr, depth + 1))
        
        return True
    
    def _luw_rule(d):
        rule = {}
        if d >= 2:
            rule[2] = (1, 1)
        if d >= 3:
            rule[3] = (2, 1)
        if d >= 4:
            rule[4] = (1, 2)
        for c in range(5, d + 1):
            rule[c] = (c - 2, c - 1)
        return rule
    
    def _construct_luw(q, d):
        rule = _luw_rule(d)
        index = {}
        order = []
        edges = []
        
        start = (0, (None,) + (0,) * d)
        index[start] = 0
        order.append(start)
        
        for i in range(len(order)):
            vertex = order[i]
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = (x[c] + y[a] * x[b]) % q
                    else:
                        y[c] = (x[c] - x[a] * y[b]) % q
                
                neighbor = (1 - side, tuple(y))
                
                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)
                
                if side == 0:
                    edges.append((index[vertex], index[neighbor]))
        
        return len(order), edges
    
    best_n = None
    best_edges = None
    
    for d in range(4, 11):
        if time.time() > deadline:
            break
        
        n, edges = _construct_luw(q, d)
        
        # Verify 3-regular
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(deg[i] != 3 for i in range(n)):
            continue
        
        if _check_girth_fast(n, edges, g):
            best_n = n
            best_edges = edges
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    n, edges = _construct_luw(q, 6)
    return n, edges
