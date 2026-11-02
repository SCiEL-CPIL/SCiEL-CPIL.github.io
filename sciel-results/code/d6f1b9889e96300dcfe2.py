import time
import sys
from collections import defaultdict, deque
import random

# EVOLVE-BLOCK-START

def _validate_graph(k, g, n, edges):
    """Validate that the graph is k-regular, simple, and has girth >= g."""
    if n <= 0:
        return False
    degree = [0] * n
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
        degree[u] += 1
        degree[v] += 1
    
    for d in degree:
        if d != k:
            return False
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check girth using BFS from each vertex (non-backtracking)
    for start in range(n):
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1))
        
        while queue:
            node, prev, dist = queue.popleft()
            if dist >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start and dist + 1 < g:
                    return False
                if (nb, node) not in visited:
                    visited.add((nb, node))
                    queue.append((nb, node, dist + 1))
    
    return True

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    # For girth g, we need 2d+1 >= g => d >= (g-1)/2
    d = (g - 1) // 2
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule for LUW
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

def _check_circulant_girth(n, a, b, g):
    """Check if circulant graph C_n({±a, ±b, n/2}) has girth >= g."""
    m = n // 2
    S = [a, n-a, b, n-b, m]
    
    # Check for cycles of length L < g
    # A cycle corresponds to a sequence s_1...s_L in S such that sum s_i ≡ 0 mod n
    # and the path doesn't backtrack (s_i != -s_{i-1}).
    # For small L, we can enumerate.
    
    # L=3: s1+s2+s3=0
    # Possibilities:
    # 1. a+a+a=0 => 3a=0
    # 2. a+a+b=0 => 2a+b=0
    # 3. a+b+b=0 => a+2b=0
    # 4. a+b+m=0 => a+b+m=0
    # 5. a+a+m=0 => 2a+m=0
    # 6. b+b+m=0 => 2b+m=0
    # 7. a+m+m=0 => a+2m=0 => a=0 (impossible)
    # 8. b+m+m=0 => b=0 (impossible)
    # 9. m+m+m=0 => 3m=0 => 3n/2=0 => 3n=0 mod n => 0=0 (Always true? No, m+m+m = 3n/2. If n is even, 3n/2 is not 0 mod n unless n=0. Wait. m = n/2. 3m = 3n/2. 3n/2 mod n = n/2 != 0. So 3-cycle with 3 m's is not 0.)
    
    # Actually, simpler: just check all sums of length L < g.
    # For L=3:
    for s1 in S:
        for s2 in S:
            if s2 == n - s1: continue # Backtrack
            for s3 in S:
                if s3 == n - s2: continue
                if (s1 + s2 + s3) % n == 0:
                    return False
    # For L=4:
    for s1 in S:
        for s2 in S:
            if s2 == n - s1: continue
            for s3 in S:
                if s3 == n - s2: continue
                for s4 in S:
                    if s4 == n - s3: continue
                    if (s1 + s2 + s3 + s4) % n == 0:
                        # Check if it's a valid cycle (not just back and forth)
                        # The non-backtracking condition ensures this.
                        return False
    # For L=5 to g-1:
    # This is getting expensive. We can use a set of reachable states.
    # State: (current_pos, last_step)
    # Start at 0, last_step = -1
    # Expand for L steps.
    
    visited = set()
    # (pos, last)
    visited.add((0, -1))
    
    for step in range(1, g):
        new_visited = set()
        for pos, last in visited:
            for s in S:
                if s == n - last: continue
                new_pos = (pos + s) % n
                if new_pos == 0 and step + 1 < g:
                    return False
                if (new_pos, s) not in visited:
                    new_visited.add((new_pos, s))
        visited = new_visited
        if not visited:
            break
            
    return True

def _circulant_search(k, g, max_n, deadline):
    if k != 5:
        return None
        
    best_n = float('inf')
    best_edges = None
    
    # Moore bound lower limit
    min_n = 426
    
    for n in range(min_n, min(max_n, 1200) + 1, 2):
        if time.time() > deadline:
            break
        m = n // 2
        for a in range(1, m):
            if time.time() > deadline: break
            for b in range(a + 1, m):
                if a == b or b == m: continue
                # Quick filter: 3-cycles
                if (3*a) % n == 0: continue
                if (3*b) % n == 0: continue
                if (2*a + b) % n == 0: continue
                if (a + 2*b) % n == 0: continue
                if (a + b + m) % n == 0: continue
                if (2*a + m) % n == 0: continue
                if (2*b + m) % n == 0: continue
                if (a + b - m) % n == 0: continue # a+b = m
                if (a - b + m) % n == 0: continue
                if (b - a + m) % n == 0: continue
                
                if _check_circulant_girth(n, a, b, g):
                    edges = []
                    for i in range(n):
                        for s in [a, n-a, b, n-b, m]:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    # Try Circulant Search first (likely to find smaller graphs if they exist)
    result = _circulant_search(k, g, 1200, deadline)
    if result is not None:
        n, edges = result
        if _validate_graph(k, g, n, edges):
            return n, edges
            
    # Fallback to LUW
    try:
        n_luw, edges_luw = _luw_graph(k, g)
        if _validate_graph(k, g, n_luw, edges_luw):
            return n_luw, edges_luw
    except Exception:
        pass
        
    # If LUW fails, try a random construction or raise
    raise Exception("Failed to construct valid graph")

# EVOLVE-BLOCK-END
