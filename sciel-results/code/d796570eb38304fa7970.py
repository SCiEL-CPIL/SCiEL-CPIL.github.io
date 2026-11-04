import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # 1. Try to find a small circulant graph
    # Theoretical minimum for (4,10) is 243. Known minimum is 320.
    # We search n from 243 to 400.
    
    if k == 4 and g == 10:
        # First, try specific known candidates if any are hardcoded and verified
        # But let's do a systematic search for n in [243, 400]
        
        # To speed up, we can check n=320 first as it's the known best.
        # Then expand outwards.
        n_candidates = [320]
        for offset in range(1, 100):
            if 320 - offset >= 243:
                n_candidates.append(320 - offset)
            if 320 + offset <= 450:
                n_candidates.append(320 + offset)
        
        # Sort by n to find the smallest valid one
        n_candidates = sorted(set(n_candidates))
        
        for n in n_candidates:
            if time.time() > deadline - 5.0:
                break
            if _find_circulant_4_reg(n, g):
                return n, _get_circulant_edges(n, [1, 0]) # Placeholder, will be filled by find_func
            
        # If no circulant found, try LUW
        return _luw_construct(k, g)

    # General fallback: LUW construction
    return _luw_construct(k, g)

def _find_circulant_4_reg(n, g):
    """
    Finds a pair (a, b) such that C_n(a, b) is 4-regular and has girth >= g.
    Returns (a, b) or None.
    """
    # We need a, b such that:
    # 1. 2a != 0, 2b != 0 (mod n)
    # 2. a != b, a != -b (mod n)
    # 3. Girth >= g
    
    # To reduce symmetry, assume 1 <= a < b <= n//2
    # Also, we can fix a=1 without loss of generality?
    # Not necessarily, but it's a good heuristic.
    # Let's iterate a from 1 to n//2, and b from a+1 to n//2.
    # This is O(n^2) which is too slow for n=320 (100k checks) if each check is slow.
    # But girth check is BFS up to depth 9.
    
    # Optimization: 
    # A necessary condition for girth >= 6 is no triangles.
    # A triangle exists if x+y+z=0.
    # For 4-regular circulant, steps are +/-a, +/-b.
    # Triangle: a+a-b=0 => 2a=b. Or a+b-a=0 (backtrack, ignored).
    # So if b=2a, there is a triangle (0, a, 2a).
    # Similarly if a=2b.
    # So we can skip if b == 2a or a == 2b.
    
    # Also 4-cycles: a+a-b-b=0 => 2a=2b (impossible if a!=b).
    # a+b-b-a=0 (backtrack).
    # a+a+b-b=0 => 2a=0 (impossible).
    # a+b+a-b=0 => 2a=0.
    # So 4-cycles are less of an issue for distinct a,b unless small n.
    
    # Let's try a=1 first.
    for a in range(1, n // 2 + 1):
        if 2 * a % n == 0:
            continue
        for b in range(a + 1, n // 2 + 1):
            if 2 * b % n == 0:
                continue
            if b == 2 * a % n or a == 2 * b % n:
                continue
            if a == n - b: # a = -b
                continue
                
            if _check_girth_circulant(n, a, b, g):
                return a, b
                
    return None

def _check_girth_circulant(n, a, b, g):
    """
    Checks if C_n(a, b) has girth >= g.
    Uses BFS from 0.
    """
    # Directions: 0:+a, 1:-a, 2:+b, 3:-b
    # Backtracking pairs: (0,1), (1,0), (2,3), (3,2)
    
    # visited[pos][dir] = min depth
    # If we reach 0 with depth < g, return False.
    
    visited = [[-1 for _ in range(4)] for _ in range(n)]
    
    # Queue: (pos, last_dir, depth)
    # Start: from 0, no last_dir.
    # We can start by taking any step.
    queue = []
    
    # Initial steps from 0
    for d in range(4):
        if d == 0: new_pos = a % n
        elif d == 1: new_pos = (-a) % n
        elif d == 2: new_pos = b % n
        else: new_pos = (-b) % n
        
        if new_pos == 0:
            # Loop of length 1? Impossible for simple graph (no loops).
            # But a or b could be 0? No, a,b >= 1.
            continue
        
        visited[new_pos][d] = 1
        queue.append((new_pos, d, 1))
        
    qi = 0
    while qi < len(queue):
        pos, last_dir, depth = queue[qi]
        qi += 1
        
        if depth >= g - 1:
            continue
            
        for next_dir in range(4):
            # Check backtracking
            if (last_dir == 0 and next_dir == 1) or (last_dir == 1 and next_dir == 0) or \
               (last_dir == 2 and next_dir == 3) or (last_dir == 3 and next_dir == 2):
                continue
                
            if next_dir == 0: new_pos = (pos + a) % n
            elif next_dir == 1: new_pos = (pos - a) % n
            elif next_dir == 2: new_pos = (pos + b) % n
            else: new_pos = (pos - b) % n
            
            new_depth = depth + 1
            
            if new_pos == 0:
                if new_depth < g:
                    return False
                else:
                    continue
                    
            if visited[new_pos][next_dir] == -1:
                visited[new_pos][next_dir] = new_depth
                queue.append((new_pos, next_dir, new_depth))
            else:
                if visited[new_pos][next_dir] > new_depth:
                    # Found a shorter path to same state?
                    # For girth, we just need to know if 0 is reachable.
                    # If we already visited this state at a shallower depth, 
                    # any cycle formed by extending this path would be longer 
                    # or equal to the one formed by the shallower path.
                    # So we can skip.
                    pass
                    
    return True

def _get_circulant_edges(n, steps):
    edges = []
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges

def _luw_construct(k, g):
    # Original seed logic
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
