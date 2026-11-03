import time
import random
import sys
from collections import defaultdict, deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Moore bound for k=3, g=16
    # n >= 1 + k * sum_{i=0}^{g/2 - 1} (k-1)^i
    # For g=16, g/2 = 8. Sum i=0 to 7 of 2^i = 2^8 - 1 = 255.
    # n >= 1 + 3 * 255 = 766? 
    # Wait, Moore bound formula:
    # If g is even, n >= 1 + k * sum_{i=0}^{g/2 - 1} (k-1)^i
    # g=16 -> g/2 = 8. Sum i=0..7.
    # 1 + 3 * (2^8 - 1) = 1 + 3 * 255 = 766.
    # Wait, the prompt says "best lower bound is 512".
    # Let's re-calculate.
    # Tree of depth 7 from a vertex.
    # Level 0: 1
    # Level 1: 3
    # Level 2: 3*2=6
    # ...
    # Level 7: 3*2^6 = 192.
    # Total vertices in tree: 1 + 3*(1 + 2 + ... + 2^6) = 1 + 3*(2^7 - 1) = 1 + 3*127 = 382.
    # This is for a *vertex*.
    # The Moore bound is the minimum number of vertices in a graph with min degree k and girth g.
    # For even girth g=2r, the bound is 1 + k * sum_{i=0}^{r-1} (k-1)^i.
    # r = 8.
    # Sum = 2^8 - 1 = 255.
    # n >= 1 + 3 * 255 = 766.
    # Why does the prompt say 512?
    # Maybe the prompt's "512" is a different bound or I am misremembering the formula.
    # Actually, the Moore bound for (3, 16) is indeed 766?
    # Let's check online knowledge: n(3, 16) lower bound.
    # Known: n(3, 8) = 14 (Heawood). Bound: 1+3*(1+2)=10.
    # n(3, 10) = 34? Bound: 1+3*(1+2+4)=16.
    # n(3, 12) = 112? Bound: 1+3*(1+2+4+8)=26.
    # n(3, 14) = ?
    # n(3, 16) = ?
    # The prompt says "best lower bound is 512". I will use 512 as the start.
    
    best_n = None
    best_edges = None
    
    # Try heuristic search for N < 1280
    # We know LUW gives ~1280.
    # We try to beat it.
    
    def generate_random_3reg(n):
        """Generate a random simple 3-regular graph on n vertices."""
        if n % 2 != 0:
            return None
        if n < 4:
            return None
        
        degrees = [0] * n
        edges = set()
        # Use a list of candidates for efficient selection
        # Simple approach: pick random pairs until degrees are met.
        # This can be slow for large n, but n < 1300 is manageable.
        
        attempts = 0
        max_attempts = 10000
        while any(d < 3 for d in degrees) and attempts < max_attempts:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if (u, v) in edges or (v, u) in edges:
                continue
            if degrees[u] >= 3 or degrees[v] >= 3:
                continue
            
            edges.add((u, v))
            degrees[u] += 1
            degrees[v] += 1
            attempts += 1
        
        if any(d != 3 for d in degrees):
            return None
            
        return list(edges)

    def get_girth(edges, n, max_girth=16):
        """Calculate the girth of the graph. Returns infinity if girth > max_girth."""
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = float('inf')
        
        for start in range(n):
            # BFS
            dist = {start: 0}
            parent = {start: -1}
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= max_girth // 2:
                    continue
                    
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and v != start:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # But we need to be careful with BFS tree.
                        # Standard girth check:
                        # If we encounter a visited node v (not parent), 
                        # we have a cycle.
                        # The length is dist[u] + dist[v] + 1.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth <= g:
                                return min_girth
            # Optimization: if we found a cycle <= g, we can stop early
            if min_girth <= g:
                return min_girth
                
        return min_girth

    def try_construct_for_n(n, time_limit):
        """Try to find a 3-regular graph with girth >= g on n vertices."""
        start_t = time.time()
        best_girth = 0
        best_edges = None
        
        for _ in range(50): # 50 restarts
            if time.time() > start_t + time_limit:
                break
                
            edges = generate_random_3reg(n)
            if not edges:
                continue
                
            cur_girth = get_girth(edges, n, g)
            if cur_girth >= g:
                return edges, cur_girth
                
            if cur_girth > best_girth:
                best_girth = cur_girth
                best_edges = edges
                
            # Local search: try to improve girth
            # We can skip this for now to keep it simple, 
            # but random restarts are often sufficient for small N.
            # However, to be effective, we should try swaps.
            
            # Simple swap heuristic
            if best_edges:
                # Try 100 random swaps
                for _ in range(100):
                    if time.time() > start_t + time_limit:
                        break
                    # Pick two random edges
                    e1 = random.choice(best_edges)
                    e2 = random.choice(best_edges)
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    # Try swapping to (u1, u2) and (v1, v2)
                    # Check validity
                    new_edges = set(best_edges)
                    new_edges.discard(e1)
                    new_edges.discard(e2)
                    
                    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                        continue
                        
                    e_new1 = (u1, u2)
                    e_new2 = (v1, v2)
                    
                    if e_new1 in new_edges or e_new2 in new_edges:
                        continue
                    if e_new1 == e_new2:
                        continue
                        
                    new_edges.add(e_new1)
                    new_edges.add(e_new2)
                    
                    new_girth = get_girth(list(new_edges), n, g)
                    if new_girth > cur_girth:
                        best_edges = list(new_edges)
                        cur_girth = new_girth
                        if cur_girth >= g:
                            return best_edges, cur_girth

        if best_girth >= g:
            return best_edges, best_girth
        return None, 0

    # Search for N
    # Start from 512, step 2 (must be even for 3-reg)
    # We want to beat 1280.
    # We try a few values.
    # To save time, we can binary search or just try a few promising values.
    # Let's try N in [512, 1200] with step 10?
    # Or just try a few: 512, 600, 700, 800, 900, 1000, 1100, 1200.
    
    target_ns = [512, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200]
    
    for n in target_ns:
        if time.time() > deadline - 10.0:
            break
        if n >= 1280:
            break
            
        # Allocate time based on remaining time
        remaining = deadline - time.time()
        time_per_n = min(5.0, remaining / 2.0) # Max 5s per N, at least 2s left
        
        edges, girth = try_construct_for_n(n, time_per_n)
        if edges:
            return n, edges
            
    # Fallback to LUW
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
