import time
import math
import random
from collections import deque

_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _deadline_exceeded(limit=140):
    return _elapsed() > limit

def _check_girth(adj, n, g):
    """
    Checks if the graph has girth at least g.
    Returns True if girth >= g, False otherwise.
    """
    # For each vertex, check if there is a cycle of length < g
    # We can do BFS from each vertex up to depth g-1
    # If we find a neighbor already visited at depth < g-1, we have a cycle.
    # Actually, simpler:
    # For each edge (u, v), remove it, find shortest path between u and v.
    # If dist < g-1, then cycle length = dist + 1 < g.
    # This is O(E * (V+E)) which is 1250 * 6250 ~ 7.8M ops. Fast enough.
    
    # Optimized: BFS from each node, stop at depth g-1.
    # If we encounter a node already in the current BFS tree (not parent),
    # and the path length < g, we have a cycle.
    
    for u in range(n):
        # BFS from u
        dist = [-1] * n
        parent = [-1] * n
        dist[u] = 0
        q = deque([u])
        
        while q:
            curr = q.popleft()
            d_curr = dist[curr]
            
            if d_curr >= g - 1:
                continue
                
            for v in adj[curr]:
                if dist[v] == -1:
                    dist[v] = d_curr + 1
                    parent[v] = curr
                    q.append(v)
                else:
                    # Found a cycle?
                    # If v is not the parent of curr, and dist[v] + dist[curr] + 1 < g?
                    # No, this logic is tricky for simple cycle detection.
                    # Better: Just check if any neighbor v has dist[v] < g-1 and v != parent[curr]
                    # and the path back is short.
                    pass
        
        # This BFS approach is prone to errors in cycle length calculation.
        # Let's use the edge-removal method for small n.
        pass
        
    # Let's use a simpler, robust method for small n:
    # For each vertex u, for each pair of neighbors v1, v2:
    # Find shortest path between v1 and v2 not using u.
    # If length < g-2, cycle length < g.
    # n=1250, deg=5. Pairs per vertex = 10.
    # Total pairs = 12500.
    # BFS for each pair: O(V+E) = 6250.
    # Total 12500 * 6250 = 78M ops. Might be too slow in Python (1-2s).
    # We have 150s limit. Should be fine.
    
    for u in range(n):
        neighbors = adj[u]
        if len(neighbors) < 2:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                v1 = neighbors[i]
                v2 = neighbors[j]
                
                # BFS from v1 to v2, excluding u
                # Limit depth to g-3
                visited = {v1: 0}
                queue = deque([(v1, -1)])
                found = False
                
                while queue:
                    curr, par = queue.popleft()
                    d = visited[curr]
                    
                    if d >= g - 3:
                        break
                        
                    for nxt in adj[curr]:
                        if nxt == par:
                            continue
                        if nxt == u:
                            continue
                        if nxt not in visited:
                            visited[nxt] = d + 1
                            if nxt == v2:
                                # Cycle length = d + 1 + 2
                                # We found path of length d+1
                                # Total cycle = d + 1 + 2 = d + 3
                                if d + 3 < g:
                                    return False
                                found = True
                                queue.clear()
                                break
                            if d + 1 < g - 3:
                                queue.append((nxt, curr))
                
                if found:
                    # If found_dist + 2 < g, return False
                    # We already checked inside loop
                    pass
                    
    return True

def construct_lw(k, g):
    q = k
    # Try minimal d
    # girth approx 2d+1
    min_d = (g - 1) // 2
    if min_d < 3:
        min_d = 3
        
    for d in range(min_d, 10):
        if _deadline_exceeded():
            break
            
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
        adj = [[] for _ in range(q ** d * 2)] # Pre-allocate if possible, but size unknown yet
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        n = len(order)
        
        # Build adj list for girth check
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check girth
        if _check_girth(adj, n, g):
            return n, edges
            
    # Fallback: return the largest d tried (safest)
    # Re-run with max d
    d = 9 # High enough
    # ... (re-implement or just return last known valid)
    # For simplicity, if loop finishes, return the last generated graph
    # But we need to ensure it's valid.
    # The loop breaks on deadline.
    # If no valid graph found, return the one with highest d (most likely valid)
    
    # Re-generate with d=5 for k=5
    if k == 5:
        d = 5
        # ... (copy logic)
        pass
        
    return None

def construct(k, g):
    global _start_time
    _start_time = time.time()
    
    result = construct_lw(k, g)
    if result is not None:
        return result
        
    # Fallback to simple construction if LW fails
    # But LW should not fail for k=5, g=9 with d=5
    raise Exception("Construction failed")

construct_lw(5, 9)
