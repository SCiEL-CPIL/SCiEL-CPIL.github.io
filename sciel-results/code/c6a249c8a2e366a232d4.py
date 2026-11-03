import time
import sys

def _check_girth(n, steps, min_girth):
    """
    Check if the circulant graph C_n(steps) has girth at least min_girth.
    steps: list of positive integers < n/2.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        for s in steps:
            j1 = (i + s) % n
            j2 = (i - s) % n
            adj[i].append(j1)
            adj[i].append(j2)
            
    # Remove duplicates and self loops (should not exist for valid steps)
    for i in range(n):
        adj[i] = list(set(adj[i]))
        
    # BFS from 0 to find shortest cycle
    # We can do BFS from all nodes, but for circulant, symmetry helps?
    # Just do BFS from 0 and 1 to be safe?
    # Actually, girth is the min over all vertices.
    # For circulant, we can just check cycles starting from 0.
    # A cycle in a circulant graph corresponds to a sum of steps = 0.
    # We want the shortest sequence of steps (with signs) that sums to 0 mod n.
    # This is the "shortest vector" problem in a lattice.
    
    # BFS on the Cayley graph of Z_n with generators +/- steps.
    # Distance from 0 to 0 is 0.
    # We want the shortest non-trivial path from 0 to 0.
    # This is equivalent to finding the shortest cycle containing 0.
    # In a vertex-transitive graph, the girth is the shortest cycle containing any vertex.
    
    # BFS
    from collections import deque
    dist = [-1] * n
    parent = [-1] * n
    q = deque()
    dist[0] = 0
    q.append(0)
    
    min_cycle = float('inf')
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
            else:
                # If v is already visited, we found a cycle
                # The length is dist[u] + dist[v] + 1?
                # No, this finds the shortest path between u and v.
                # To find the shortest cycle, we need to be careful.
                # Standard BFS for girth:
                # When we see an edge (u, v) where dist[v] != -1 and v != parent[u],
                # the cycle length is dist[u] + dist[v] + 1.
                # But this only finds cycles that go through the root?
                # No, BFS from a single root finds the shortest cycle *containing the root*.
                # In a vertex-transitive graph, the girth is the length of the shortest cycle containing any vertex.
                # So BFS from 0 is sufficient for circulant graphs.
                
                if v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        
    return min_cycle >= min_girth

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    # Moore bound
    r = g // 2
    moore_bound = 2 * sum((k-1)**i for i in range(r))
    
    best_n = float('inf')
    best_edges = []
    
    # Search for circulant graphs
    # n must be even
    # We search a range of n
    # If k=4, g=10, moore_bound=242.
    # We try n from 242 to 500.
    
    for n in range(moore_bound, 600, 2):
        if time.time() > deadline:
            break
            
        # Steps s1, s2
        # 1 <= s1 < s2 < n/2
        for s1 in range(1, n // 2):
            for s2 in range(s1 + 1, n // 2):
                # Quick 4-cycle check
                if 2*s1 % n == 0 or 2*s2 % n == 0 or (s1+s2) % n == 0 or (s1-s2) % n == 0:
                    continue
                
                if _check_girth(n, [s1, s2], g):
                    edges = []
                    for i in range(n):
                        for s in [s1, s2]:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges
                        
    # Fallback to LUW
    # (Copy of original logic)
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
