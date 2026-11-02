"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph with girth at least g, minimizing n."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # For the specific case k=3, g=16
    # Moore bound: 1 + k * sum_{i=0}^{(g-2)/2 - 1} (k-1)^i = 1 + 3*sum_{i=0}^{6} 2^i = 1+3*127 = 382
    # Actually for girth 16 (even), radius is g/2 - 1 = 7
    # Moore bound for k-regular girth 2r: 2 * sum_{i=0}^{r-1} (k-1)^i
    # For k=3, r=7: 2 * (1+2+4+8+16+32+64) = 2*127 = 256
    # Wait, let me recalculate. For even girth g=2r, the Moore bound is 2*sum_{i=0}^{r-1}(k-1)^i
    # For k=3, r=8 (since g=16, r=g/2=8): 2*(1+2+4+8+16+32+64+128) = 2*255 = 510
    # So Moore bound is 510, and smallest known is 936.
    
    # Strategy: Use the Lazebnik-Ustimenko-Woldar (LUW) construction as a baseline,
    # then try to improve via edge swaps if time permits.
    # But first, let's try to find a better explicit construction.
    
    # The LUW construction CD(d, q) gives a q-regular bipartite graph with girth d+5 (odd d)
    # or d+4 (even d). For k=3, g=16: we need girth >= 16.
    # If d is odd, girth = d+5 >= 16 => d >= 11. Smallest odd d >= 11 is 11, girth = 16.
    # The number of vertices in CD(d, q) is 2*q^d. For q=3, d=11: 2*3^11 = 2*177147 = 354294.
    # That's way too many. We need ~936.
    
    # The LUW construction is far from optimal. We need a different approach.
    
    # Approach: Use a random 3-regular graph with n near the Moore bound, then use
    # local search (edge swaps) to increase girth.
    
    # Moore bound for k=3, g=16 is 510. Smallest known is 936.
    # Let's try n values from 510 to 936, looking for a valid graph.
    
    # Actually, for a 3-regular graph with girth 16, we need n >= 510.
    # Let's try to construct such a graph using a systematic approach.
    
    # Key insight: For 3-regular graphs with high girth, the best constructions
    # use algebraic methods. The known 936-vertex graph uses a specific algebraic construction.
    
    # Let me try a different approach: use the known fact that for k=3, g=16,
    # the optimal is 936 vertices. I'll try to construct it directly.
    
    # Actually, let's use a practical approach: generate random 3-regular graphs
    # and check girth, using edge swaps to improve.
    
    def moore_bound(k, g):
        if g % 2 == 0:
            r = g // 2
            return 2 * sum((k-1)**i for i in range(r))
        else:
            r = (g - 1) // 2
            return 1 + k * sum((k-1)**i for i in range(r))
    
    mb = moore_bound(k, g)
    
    # Try to construct a graph with n vertices
    def try_construct(n, k, g, time_limit):
        """Try to construct a k-regular graph on n vertices with girth >= g."""
        # Generate a random k-regular graph using the configuration model
        # Then check girth and try to improve via edge swaps
        
        # For 3-regular graphs, use the following approach:
        # 1. Create 3n half-edges
        # 2. Randomly pair them to form edges
        # 3. Check for simplicity (no loops, no multi-edges)
        # 4. Check girth
        # 5. If girth < g, try edge swaps to increase girth
        
        # This is complex. Let's use a simpler approach for now.
        
        # Use a known construction or return the LUW result if it's valid.
        
        # For the specific case k=3, g=16, let's try the LUW construction
        # with appropriate parameters, or use a different method.
        
        # Actually, let's just use the LUW construction as a fallback.
        # It's not optimal but it's valid.
        
        pass
    
    # Let's use the original LUW construction as a baseline
    q = k
    # For g=16, we need girth >= 16
    # LUW: CD(d, q) has girth d+5 (odd d) or d+4 (even d)
    # For odd d: d+5 >= 16 => d >= 11
    # For even d: d+4 >= 16 => d >= 12
    # Let's use d=11 (odd), girth=16
    d = 11 if (g - 5) % 2 else 12
    # Actually, let's compute d properly
    # We want girth >= g
    # If d is odd: girth = d+5, so d >= g-5, and d must be odd
    # If d is even: girth = d+4, so d >= g-4, and d must be even
    # Let's pick the smallest valid d
    if (g - 5) % 2 == 1:
        d = g - 5  # odd
    else:
        d = g - 4  # even
    # Ensure d >= 3
    d = max(3, d)
    # Make sure d has the right parity
    if d % 2 == 1 and (g - 5) % 2 != 1:
        d += 1
    if d % 2 == 0 and (g - 4) % 2 != 0:
        d += 1
    
    # Recompute: for g=16
    # g-5 = 11 (odd), so d=11 works, girth = 16
    # g-4 = 12 (even), so d=12 works, girth = 16
    # Let's use d=11
    if (g - 5) % 2 == 1:
        d = max(3, g - 5)
    else:
        d = max(3, g - 4)
    
    # For q=3, d=11: 2*3^11 = 354294 vertices. Too many.
    # The LUW construction is not suitable for small graphs.
    
    # Let's try a completely different approach.
    # Use a random 3-regular graph and check girth.
    
    def generate_random_3_regular(n):
        """Generate a random 3-regular simple graph on n vertices."""
        # n must be even for 3-regular graph
        if n % 2 != 0:
            return None
        # Use the configuration model
        # Each vertex has 3 stubs
        stubs = []
        for v in range(n):
            for _ in range(3):
                stubs.append(v)
        random.shuffle(stubs)
        # Pair stubs to form edges
        edges = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                continue  # skip loops
            if (u, v) in edges or (v, u) in edges:
                continue  # skip multi-edges
            edges.add((min(u, v), max(u, v)))
        # Check if we have a valid 3-regular graph
        if len(edges) != n * 3 // 2:
            return None
        # Check degree
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if any(deg != 3 for deg in degree):
            return None
        return list(edges)
    
    def compute_girth(n, edges):
        """Compute the girth of a graph using BFS from each vertex."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = float('inf')
        for start in range(n):
            # BFS from start, tracking distance and parent
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Found a cycle
                        # The cycle length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth if min_girth != float('inf') else n  # no cycle found
    
    # Try to find a valid graph
    best_n = None
    best_edges = None
    
    # Start from Moore bound and go up
    # For k=3, g=16: mb=510, known best=936
    # Let's try a range
    n_start = mb
    n_end = min(mb + 500, 1000)
    
    # This approach is too slow. Let's use a different strategy.
    
    # Use the known 936-vertex construction if available, or fall back to LUW.
    
    # For now, let's just use the LUW construction with the smallest valid parameters.
    # It's not optimal but it's valid.
    
    q = k
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
# EVOLVE-BLOCK-END
