import random
import time
from collections import deque

def construct(k, g):
    # If the specific target is not (3, 16), fall back to the original LUW construction
    if k != 3 or g != 16:
        return _construct_luw(k, g)
    
    # Try to find a small circulant (3, 16)-cage
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Moore bound for (3, 16) is 512
    n_min = 512
    n_max = 1200
    
    # We search for n in this range, preferring even n
    for n in range(n_min, n_max + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Skip odd n for 3-regular graphs (handshake: n*k must be even, so n must be even for k=3)
        if n % 2 != 0:
            continue
            
        # Systematic search over generator pairs (b, c) with a=1
        # To reduce symmetry: assume 1 <= b < c <= n-1, and b != n-b, c != n-c, b != n-c, etc.
        # We'll do a random search with more trials for smaller n
        
        max_trials = 5000
        trials = 0
        found = False
        
        while trials < max_trials and not found:
            if time.time() - start_time > time_limit:
                break
            trials += 1
            
            b = random.randint(2, n - 2)
            c = random.randint(2, n - 2)
            
            # Ensure 1, b, c and their negatives are all distinct mod n
            dirs = [1, -1, b, -b, c, -c]
            if len(set(d % n for d in dirs)) < 6:
                continue
                
            # Check girth using optimized BFS
            if _check_girth_circulant(n, [1, b, c], g):
                # Construct edges
                edges = []
                seen = set()
                for i in range(n):
                    for a in [1, b, c]:
                        j = (i + a) % n
                        u, v = min(i, j), max(i, j)
                        if (u, v) not in seen:
                            seen.add((u, v))
                            edges.append((u, v))
                return n, edges
                found = True
                
    # Fallback to LUW if no circulant found
    return _construct_luw(k, g)

def _check_girth_circulant(n, gens, g):
    """
    Check if the circulant graph with generators gens on Z_n has girth >= g.
    Returns True if girth >= g, False otherwise.
    Uses BFS from vertex 0. The girth of a vertex-transitive graph equals the girth from any vertex.
    """
    # For a 3-regular graph, girth >= g means no cycle of length < g.
    # BFS from 0: if we find a non-tree edge (u,v) where u is at depth d and v is at depth d',
    # the cycle length is d + d' + 1. We need all such cycles to have length >= g.
    # We only need to explore up to depth (g-1)//2 since a cycle of length L has two vertices
    # at distance floor(L/2) from each other... actually let's think more carefully.
    
    # A cycle of length L < g: in BFS from 0, the two endpoints of any edge on the cycle
    # that are at the maximum depth will have depths that sum to at most L-1.
    # To detect all cycles of length < g, we need to explore to depth floor((g-1)/2).
    # Actually, if there's a cycle of length L, then there exist two vertices on it at
    # distance floor(L/2) from each other. In BFS, one will be at depth d and the other at
    # depth d' with |d-d'| <= 1 and d+d' >= L-1... this is getting complicated.
    
    # Simpler: just do full BFS and check all non-tree edges.
    # For n up to ~1200 and 6 neighbors per vertex, this is fast.
    
    dist = [-1] * n
    parent = [-1] * n  # parent in BFS tree
    dist[0] = 0
    queue = deque([0])
    
    while queue:
        u = queue.popleft()
        d = dist[u]
        
        for a in gens:
            for sign in [1, -1]:
                v = (u + sign * a) % n
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # v is already visited. Check if this edge creates a short cycle.
                    # Skip if v is the parent of u (backtracking edge)
                    if v == parent[u]:
                        continue
                    # The cycle length is dist[u] + dist[v] + 1
                    # This is valid because the BFS tree path from 0 to u plus the edge (u,v)
                    # plus the BFS tree path from v to 0 forms a closed walk that contains
                    # a cycle of length at most dist[u] + dist[v] + 1.
                    # Actually, it's exactly a cycle if the paths from 0 to u and 0 to v
                    # only share the vertex 0. In a BFS tree of a simple graph, this is
                    # guaranteed for non-tree edges where neither is the parent of the other.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _construct_luw(k, g):
    # Original LUW construction - produces a 3-regular graph with girth at least g
    # This is a known construction but may not be optimal
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
