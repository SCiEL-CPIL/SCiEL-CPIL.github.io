import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 140 seconds wall clock limit (10s safety margin)
    
    if k != 4 or g != 10:
        # Fallback to LUW construction for other parameters
        return _luw_construct(k, g)
    
    def check_girth_fast(p, a, b, girth_req):
        """Check if Cayley graph on Z_p x Z_p with generators (1,0),(-1,0),(a,b),(-a,-b)
        has girth >= girth_req using BFS from (0,0)."""
        # Precompute neighbor offsets
        offsets = [(1, 0), (p-1, 0), (a, b), (p-a, p-b)]
        
        # BFS state: (curr_x, curr_y, prev_x, prev_y)
        # Start: neighbors of (0,0) with prev=(0,0)
        q = deque()
        visited = set()
        
        for dx, dy in offsets:
            nx, ny = dx % p, dy % p
            state = (nx, ny, 0, 0)
            q.append((nx, ny, 0, 0, 1))
            visited.add(state)
        
        while q:
            cx, cy, px, py, depth = q.popleft()
            
            # Check if we returned to origin
            if (cx, cy) == (0, 0) and depth < girth_req:
                return True  # Found short cycle
            
            # Only expand if depth + 1 < girth_req
            if depth + 1 >= girth_req:
                continue
            
            for dx, dy in offsets:
                nx, ny = (cx + dx) % p, (cy + dy) % p
                # Non-backtracking
                if (nx, ny) == (px, py):
                    continue
                state = (nx, ny, cx, cy)
                if state not in visited:
                    visited.add(state)
                    q.append((nx, ny, cx, cy, depth + 1))
        
        return False  # No short cycle
    
    def build_edges(p, a, b):
        """Build edge list for valid Cayley graph."""
        edges = []
        offsets = [(1, 0), (p-1, 0), (a, b), (p-a, p-b)]
        for x in range(p):
            for y in range(p):
                u = x * p + y
                for dx, dy in offsets:
                    nx, ny = (x + dx) % p, (y + dy) % p
                    v = nx * p + ny
                    if u < v:
                        edges.append((u, v))
        return p * p, edges
    
    # Search for valid (p, a, b)
    # Try primes starting from smallest possible
    candidates_p = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_result = None
    best_n = float('inf')
    
    for p in candidates_p:
        if time.time() > deadline - 5:
            break
        
        for a in range(1, p):
            if time.time() > deadline - 5:
                break
            for b in range(0, p):
                # Skip invalid generator combinations
                if a == 0 and b == 0:
                    continue
                if (a == 1 and b == 0) or (a == p-1 and b == 0):
                    continue
                
                if not check_girth_fast(p, a, b, g):
                    n, edges = build_edges(p, a, b)
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                        return best_result
    
    # Fallback to LUW construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """LUW (Lubotzky-Weiss) style construction as fallback."""
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
