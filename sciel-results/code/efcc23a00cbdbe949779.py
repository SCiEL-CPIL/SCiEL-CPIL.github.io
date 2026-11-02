import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if k != 3:
        return _fallback_construct(k, g)

    min_n = 256
    
    for n in range(min_n, 2000):
        if time.time() > deadline - 10:
            break
            
        attempts = 100
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
                
            a = random.randint(2, n - 1)
            b = random.randint(2, n - 1)
            if a == b or a == 1 or b == 1:
                continue
            
            S = (1, a, b)
            
            # Probabilistic check
            if _prob_has_short_cycle(n, S, 7):
                continue
                
            # Full check
            if _full_check_girth(n, S, g):
                edges = []
                for i in range(n):
                    for s in S:
                        u = i
                        v = n + (i + s) % n
                        edges.append((u, v))
                return 2 * n, edges

    return _fallback_construct(k, g)

def _prob_has_short_cycle(n, S, max_depth):
    # Sample 10 starting points
    starts = random.sample(range(n), min(10, n))
    for start in starts:
        if _bfs_find_cycle(start, n, S, max_depth):
            return True
    return False

def _bfs_find_cycle(start, n, S, max_depth):
    queue = deque()
    queue.append((start, -1, 0))
    visited = set()
    
    while queue:
        node, prev, depth = queue.popleft()
        
        if node == start and depth > 0:
            return True
            
        if depth >= max_depth:
            continue
            
        for s in S:
            next_node = (node + s) % n
            if next_node == prev:
                continue
            state = (next_node, node)
            if state in visited:
                continue
            visited.add(state)
            queue.append((next_node, node, depth + 1))
            
    return False

def _full_check_girth(n, S, target_girth):
    # Check if girth >= target_girth
    # Target girth is even.
    # We check for cycles of length < target_girth.
    # Max depth to check: target_girth // 2 - 1
    # If target_girth = 16, check depths 1..7.
    # If a cycle of length L < 16 is found, return False.
    
    max_depth = target_girth // 2 - 1
    
    # To speed up, we can check all vertices
    for start in range(n):
        if _bfs_find_cycle(start, n, S, max_depth):
            return False
            
    return True

def _fallback_construct(k, g):
    # Original LUW construction
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
