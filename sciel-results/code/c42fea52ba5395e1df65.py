"""Improved program: Circulant graph search for k=5, g=9."""
# EVOLVE-BLOCK-START
import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5-second safety margin before 120s limit
    
    # Moore bound for k=5, g=9 (odd girth)
    # n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # For g=9, (9-3)/2 = 3. Sum = 1 + 4 + 16 + 64 = 85.
    # n >= 1 + 5*85 = 426.
    # We search for even n (required for odd degree circulant with diameter element)
    
    def get_girth_circulant(n, a, b):
        """
        Calculate girth of circulant graph C_n({a, -a, b, -b, n/2}).
        Returns the length of the shortest cycle.
        """
        # BFS from 0
        # dist[v] = shortest distance from 0 to v
        # Since it's vertex transitive, girth is the shortest cycle through 0?
        # No, girth is the shortest cycle in the graph.
        # In a vertex-transitive graph, girth is the shortest cycle through any vertex.
        # So BFS from 0 is sufficient to find the girth?
        # Yes, if we detect a back edge (u, v) where v is already visited and v != parent[u],
        # the cycle length is dist[u] + dist[v] + 1.
        # We need to ensure this is a simple cycle. In a BFS tree of an unweighted graph,
        # the path from 0 to u and 0 to v are simple. Their union plus (u,v) forms a cycle.
        # It might not be the *shortest* cycle if the paths share vertices, but 
        # the shortest cycle through 0 is found by BFS.
        # Actually, the girth of the graph is the minimum over all vertices of the shortest cycle through that vertex.
        # By vertex transitivity, it is the shortest cycle through 0.
        
        dist = [-1] * n
        parent = [-1] * n
        dist[0] = 0
        queue = [0]
        qi = 0
        min_cycle = float('inf')
        
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            
            # We only need to check cycles up to length g-1
            # If dist[u] + 1 >= g, we can stop? No, a cycle could form at a deeper level
            # but we only care if it's < g.
            # Max depth needed: if a cycle of length L exists, it will be detected at depth <= L/2.
            # So if dist[u] >= (g-1)//2, any cycle formed by u will have length >= 2*dist[u] + 1 >= g.
            if dist[u] >= (g - 1) // 2:
                continue
                
            neighbors = [(u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n, (u + n // 2) % n]
            
            for v in neighbors:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Back edge
                    # Cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g:
                        return min_cycle
        return min_cycle

    def build_circulant_edges(n, a, b):
        edges = []
        half_n = n // 2
        # To avoid duplicates, only add edges (u, v) where u < v
        for u in range(n):
            v1 = (u + a) % n
            v2 = (u - a) % n
            v3 = (u + b) % n
            v4 = (u - b) % n
            v5 = (u + half_n) % n
            
            for v in [v1, v2, v3, v4, v5]:
                if u < v:
                    edges.append((u, v))
        return edges

    # Search parameters
    # n must be even.
    # a, b must be such that {a, -a, b, -b, n/2} are distinct mod n.
    # This implies 2a != 0, 2b != 0, a != +/- b, a != 0, b != 0, a != n/2, b != n/2.
    
    # Start from Moore bound
    n_min = 426
    if n_min % 2 != 0:
        n_min += 1
        
    # Limit search time
    max_n = 2000  # Safety cap, should find a solution well before this if one exists
    
    for n in range(n_min, max_n, 2):
        if time.time() > deadline - 5:
            break
            
        # Try small values for a and b
        # We need a, b in [1, n//2 - 1]
        # And a != b, a != -b (mod n), etc.
        
        found = False
        for a in range(1, n // 2):
            if time.time() > deadline - 5:
                break
            # a cannot be n/2 (already checked by range)
            for b in range(a + 1, n // 2):
                # Check distinctness
                # a, -a, b, -b, n/2
                # -a = n-a. -b = n-b.
                # Distinct if:
                # a != b, a != n-b (a+b != n), n-a != b (same), n-a != n-b (a!=b)
                # a != n/2, b != n/2 (guaranteed by range)
                # 2a != 0, 2b != 0 (guaranteed by range since n even, a < n/2)
                
                if (a + b) % n == 0:
                    continue
                if (a - b) % n == 0:
                    continue
                if (a + b) % n == n: # a+b=n is covered by (a+b)%n==0
                    continue
                
                # Also need a != -a (2a!=0) which is true.
                # Need a != b, a != -b.
                # -b = n-b. a = n-b => a+b=n.
                
                girth = get_girth_circulant(n, a, b)
                if girth >= g:
                    edges = build_circulant_edges(n, a, b)
                    return n, edges

    # Fallback: Use LUW construction if search fails
    # Re-implement LUW briefly as fallback
    q = k
    d = 5 # For k=5, g=9, d=5 gives girth 10, n=6250
    # Simple LUW implementation for fallback
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: x * y % q
    
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
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            break
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END
