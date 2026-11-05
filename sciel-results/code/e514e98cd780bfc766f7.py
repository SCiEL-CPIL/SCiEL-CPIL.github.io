import time
import random
import sys
from collections import defaultdict, deque

def construct(k, g):
    """Construct a k-regular graph with girth >= g using cycle-avoidant random construction."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Moore bound for lower limit
    moore_bound = 1 + k * (1 + (k-1) + (k-1)**2)  # rough lower bound
    if k == 3:
        moore_bound = 1 + 3 + 6 + 12  # = 22, but actual Moore bound for girth 16 is higher
        # Exact Moore bound: 1 + k * sum_{i=0}^{floor((g-1)/2)-1} (k-1)^i
        half_g = (g - 1) // 2
        moore_bound = 1 + k * sum((k-1)**i for i in range(half_g))
    
    # We'll try to construct graphs with increasing n
    # Start from a reasonable estimate
    best_n = None
    best_edges = None
    
    # Try different sizes, starting from smaller
    # For girth 16, k=3: known best is 936, Moore bound is 512
    # We'll try to find something valid, then try to minimize
    
    def try_construct(n, k, g, rng, time_limit):
        """Try to construct a k-regular graph on n vertices with girth >= g."""
        if n * k % 2 != 0:
            return None
        if n < k:
            return None
        
        # Configuration model: each vertex has k half-edges
        # We'll pair them up sequentially, checking girth constraint
        
        # Represent half-edges as (vertex, stub_index)
        half_edges = [(v, i) for v in range(n) for i in range(k)]
        rng.shuffle(half_edges)
        
        # Track which half-edges are used
        used = [False] * (n * k)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        edges = []
        
        # For girth checking, we need to detect closed non-backtracking walks
        # Strategy: maintain the graph incrementally, and after each edge addition,
        # check if any new short cycle is created
        
        # For efficiency, we'll use a different approach:
        # Build the complete graph first, then verify girth
        # But that's O(n^2) which is too slow for large n
        
        # Better: use the fact that we're building incrementally
        # After adding edge (u,v), any new cycle must use this edge
        # So we check: is there a path from u to v of length < g-1 that doesn't use (u,v)
        # and doesn't backtrack?
        
        # Actually, for a 3-regular graph being built incrementally,
        # the girth check is expensive. Let's use a different strategy.
        
        # Strategy: build a random 3-regular graph, then check girth
        # If girth is too small, try again with a different seed
        
        # For the random 3-regular graph, use the configuration model
        # with perfect matching
        
        # Generate a random perfect matching of 3n stubs
        # (3n must be even)
        if n * k % 2 != 0:
            return None
        
        # Shuffle and pair
        stubs = list(range(n * k))
        rng.shuffle(stubs)
        
        # Pair up: stubs[2i] with stubs[2i+1]
        adj = [set() for _ in range(n)]
        edges = []
        valid = True
        
        for i in range(0, n * k, 2):
            a, b = stubs[i], stubs[i + 1]
            va, vb = a // k, b // k
            if va == vb:
                # Loop, invalid
                valid = False
                break
            if vb in adj[va]:
                # Multi-edge, invalid
                valid = False
                break
            adj[va].add(vb)
            adj[vb].add(va)
            edges.append((va, vb))
        
        if not valid:
            return None
        
        # Check girth: BFS from each vertex, find shortest cycle
        # For girth >= g, we need no cycle of length < g
        # Use BFS to find shortest path from each vertex to itself (excluding trivial)
        
        # More efficient: for each edge (u,v), check if there's a path from u to v
        # of length < g-1 (not using edge (u,v))
        # If yes, then there's a cycle of length <= g-1
        
        # For girth 16, this means checking paths of length up to 14
        # This is expensive for large n
        
        # Alternative: use the fact that in a 3-regular graph,
        # we can do a BFS tree from each vertex and look for back-edges
        
        # Let's use a simpler check: BFS from each vertex, stop at depth g//2
        # If we find a vertex that's reachable via two different paths of length <= g//2,
        # we have a cycle
        
        # Actually, the cleanest way: for each vertex, do BFS up to depth (g-1)//2
        # and check for cycles
        
        half_depth = (g - 1) // 2
        
        for start in range(n):
            # BFS from start
            visited = {start: 0}
            parent = {start: None}
            queue = deque([start])
            found_cycle = False
            
            while queue:
                curr = queue.popleft()
                depth = visited[curr]
                if depth >= half_depth:
                    continue
                for nb in adj[curr]:
                    if nb == parent.get(curr):
                        continue  # Don't backtrack
                    if nb in visited:
                        # Found a cycle
                        # Cycle length = depth + 1 + (depth of nb)
                        # Actually, cycle length = visited[curr] + 1 + visited[nb] + 1?
                        # No: path from start to curr has length visited[curr]
                        # path from start to nb has length visited[nb]
                        # The cycle is: curr -> ... -> start -> ... -> nb -> curr
                        # Length = visited[curr] + 1 + visited[nb]
                        cycle_len = visited[curr] + 1 + visited[nb]
                        if cycle_len < g:
                            return None
                    else:
                        visited[nb] = depth + 1
                        parent[nb] = curr
                        queue.append(nb)
        
        # If we get here, girth >= g
        return edges
    
    # Try different n values, starting from smaller
    # For k=3, g=16: try n from moore_bound upward
    # But we need n*k to be even, so n must be even for k=3
    
    # Start from a reasonable lower bound
    # Known results: n(3,16) <= 936, lower bound 512
    # We'll try to find a valid graph, preferring smaller n
    
    # For efficiency, try a few random constructions for each n
    # and keep the best valid one found
    
    # Try n values
    n_start = max(moore_bound, 512)  # Use 512 as practical lower bound
    n_end = 1000  # Upper bound to try
    
    # We'll try n values in increasing order, but also try some random n values
    # to find a valid graph quickly, then try to minimize
    
    # First, try to find ANY valid graph
    found_any = False
    best_result = None
    
    # Try a range of n values
    # For speed, try n values that are multiples of 2 (since k=3, n must be even)
    for n in range(n_start, n_end + 1, 2):
        if time.time() > deadline - 20:  # Reserve 20s for finalization
            break
        
        # Try a few random constructions for this n
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            rng = random.Random(attempt * 1000 + n)
            result = try_construct(n, k, g, rng, deadline)
            if result is not None:
                if best_result is None or n < best_result[0]:
                    best_result = (n, result)
                    found_any = True
                break  # Found valid graph for this n, move to next n
        
        # If we found a valid graph and n is getting large, we might want to stop
        # and try to find smaller ones, but for now, keep going to find the minimum
    
    if best_result is not None:
        return best_result
    
    # If no valid graph found in the range, fall back to a known construction
    # or return a large random graph that we know is valid
    
    # Fallback: use the LUV construction from the seed
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
