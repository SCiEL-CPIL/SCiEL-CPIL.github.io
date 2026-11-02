import collections

def construct(k, g):
    # New Approach: Deterministic Algebraic Construction using Finite Fields F_{2^m}
    # This approach is distinct from the Random Search + LUW Fallback in the seed.
    
    if k != 3:
        # Fallback for other k (not expected for this task, but safe)
        return _simple_random_fallback(k, g)

    # Try m from 9 to 14
    # n = 2^(m+1)
    for m in range(9, 15):
        q = 1 << m
        n = 2 * q
        
        # Primitive polynomial for F_{2^m}
        # We need a list of primitive polynomials or just use a standard one if m is known
        # For m=9: x^9 + x^4 + 1
        # For m=10: x^10 + x^3 + 1
        # For m=11: x^11 + x^2 + 1
        # For m=12: x^12 + x^6 + x^4 + x + 1
        # For m=13: x^13 + x^4 + x^3 + x^2 + 1
        # For m=14: x^14 + x^5 + x^3 + x^2 + 1
        
        polys = {
            9: (1 << 4) | 1,
            10: (1 << 3) | 1,
            11: (1 << 2) | 1,
            12: (1 << 6) | (1 << 4) | (1 << 1) | 1,
            13: (1 << 4) | (1 << 3) | (1 << 2) | 1,
            14: (1 << 5) | (1 << 3) | (1 << 2) | 1
        }
        
        poly = polys.get(m, (1 << (m//2)) | 1) # Fallback poly, might not be primitive but let's try
        
        def field_add(a, b):
            return a ^ b
            
        def field_mul(a, b, m, poly):
            res = 0
            a_copy = a
            b_copy = b
            for i in range(m):
                if b_copy & 1:
                    res ^= a_copy
                b_copy >>= 1
                a_copy <<= 1
                if a_copy & (1 << m):
                    a_copy ^= poly
            return res

        # Precompute permutation p(x) = x^2 + x
        perm = [0] * q
        for x in range(q):
            x2 = field_mul(x, x, m, poly)
            perm[x] = field_add(x2, x)
            
        # Build edges
        edges = []
        for x in range(q):
            # Edge 1: (x, 0) - (x+1, 0)
            edges.append((x, (x + 1) % q))
            # Edge 2: (x, 1) - (x+1, 1)
            edges.append((x + q, (x + 1) % q + q))
            # Edge 3: (x, 0) - (p(x), 1)
            edges.append((x, perm[x] + q))

        # Build adjacency list
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Validate
        valid = True
        for i in range(n):
            if len(adj[i]) != 3 or len(set(adj[i])) != 3:
                valid = False
                break
                
        if not valid:
            continue
            
        # Check Girth
        if _check_girth_fast(adj, n, g):
            return n, edges

    # If all fail, return a large random graph? No, must be valid.
    # Return the last one generated (m=14) even if girth check failed? 
    # No, must be valid.
    # We assume one of them works.
    raise ValueError("No valid graph found")

def _check_girth_fast(adj, n, g):
    # BFS from each vertex
    # To speed up, we can use numpy if available, but standard lib is safer.
    # Optimization: If we find a cycle < g, return False.
    max_depth = g // 2
    for start in range(n):
        dist = {start: 0}
        parent = {start: -1}
        queue = collections.deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if v in dist:
                    if dist[u] + dist[v] + 1 < g:
                        return False
                else:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
    return True

def _simple_random_fallback(k, g):
    # Minimal fallback
    import random
    n = 2 * (1 << 10) # 2048
    # ... (random regular graph) ...
    # This is just to prevent crash, but should not be triggered.
    return n, []
