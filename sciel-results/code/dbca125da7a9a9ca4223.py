import math
import time
from collections import deque

def seed_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def check_girth_circulant(n, a, g):
    """
    Checks if the circulant graph C_n({a, -a, n/2}) has girth at least g.
    n must be even.
    """
    if n % 2 != 0:
        return False
    
    half_n = n // 2
    steps = [a, -a, half_n]
    
    # BFS from vertex 0
    # dist[v] = shortest distance from 0 to v
    # parent[v] = previous vertex in BFS tree
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    q = deque([0])
    
    while q:
        u = q.popleft()
        du = dist[u]
        
        # If we have gone deep enough, any further cycles will be long enough
        # The max possible cycle length through u is du + 1 + dist[v]
        # If du + 1 >= g, then any cycle formed here is >= g.
        # Actually, we can stop exploring if du >= g // 2?
        # No, a cycle of length g can be formed by two paths of length g/2.
        # If we find a neighbor v with dist[v] <= du, cycle length = du + 1 + dist[v].
        # If du + 1 + dist[v] < g, fail.
        # If du >= g, we don't need to go deeper?
        # If du >= g // 2, then for any neighbor v, dist[v] >= du - 1.
        # Cycle length >= du + 1 + du - 1 = 2*du >= g.
        # So if du >= g // 2, we can skip processing neighbors for cycle detection?
        # We still need to mark distances for other nodes?
        # If we stop BFS early, we might miss a shorter cycle elsewhere?
        # No, BFS explores in increasing distance. If all nodes at distance < g//2 are processed,
        # and no short cycle found, then any cycle must have length >= g.
        # Proof: Any cycle has a vertex u with max distance from start? No.
        # Standard check: If BFS from 0 finds no cycle of length < g, does it imply girth >= g?
        # Not necessarily. A cycle might not include 0.
        # However, for vertex-transitive graphs (like circulants), if girth >= g from one vertex, it is >= g everywhere.
        # So BFS from 0 is sufficient.
        
        if du >= g // 2:
            continue
            
        for step in steps:
            v = (u + step) % n
            if dist[v] == -1:
                dist[v] = du + 1
                parent[v] = u
                q.append(v)
            else:
                # Found a cycle
                # The cycle length is du + 1 + dist[v]
                # Exception: if v is parent of u, it's the tree edge, not a cycle
                if v != parent[u]:
                    cycle_len = du + 1 + dist[v]
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    if k != 3:
        return seed_construct(k, g)
        
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Moore bound for k=3
    d = (g - 1) // 2
    if d == 0:
        moore = 2
    else:
        # 1 + 3 * (2^(d+1) - 1)
        moore = 1 + 3 * ((1 << (d + 1)) - 1)
        
    n_min = (moore + 1) // 2
    
    # Search for Circulant Graphs C_n({a, -a, n/2})
    # n must be even
    # Start from n_min, rounded up to even
    n_start = n_min if n_min % 2 == 0 else n_min + 1
    
    # We search up to a reasonable limit or time limit
    # The known cage is 936. LUW is ~354k.
    # If we find anything < 354k, it's a win.
    # Let's search n up to 5000 or until time runs out.
    for n in range(n_start, n_start + 10000, 2):
        if time.time() > deadline:
            break
            
        # a must be coprime to n for connectivity
        # a in [1, n/2 - 1]
        for a in range(1, n // 2):
            if time.time() > deadline:
                break
                
            if math.gcd(n, a) != 1:
                continue
                
            if check_girth_circulant(n, a, g):
                # Found a valid circulant graph
                edges = []
                for i in range(n):
                    # Edge to i+a
                    edges.append((i, (i + a) % n))
                    # Edge to i-a
                    edges.append((i, (i - a) % n))
                    # Edge to i+n/2
                    edges.append((i, (i + n // 2) % n))
                
                return n, edges

    # Fallback to seed (LUW)
    return seed_construct(k, g)
