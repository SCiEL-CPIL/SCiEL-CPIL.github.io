<<<<<<< SEARCH
def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    best_result = None
    best_n = float('inf')
    
    q = k
    
    # Try LUW construction with varying d
    # Girth of LUW(d, q) is approx 2d+2
    # We need girth >= g
    # So 2d+2 >= g => d >= (g-2)/2
    # For g=16, d >= 7
    # But LUW(d=7, q=3) has ~2916 vertices, too many
    # LUW(d=6, q=3) has ~972 vertices but girth only 14
    # Let's try d=6 first (might have higher actual girth), then d=7
    
    min_d = max(3, (g - 2) // 2)
    
    # Try d from min_d-1 to min_d+3 (allow some flexibility)
    for d in range(max(3, min_d - 1), min_d + 4):
        if _time_left() < 10.0:
            break
            
        # Construct LUW graph
        result = _construct_luw(q, d)
        if result is None:
            continue
            
        n, edges = result
        
        # Verify girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if _check_girth_fast(n, adj, g):
            if n < best_n:
                best_n = n
                best_result = (n, edges)
            return n, edges
        else:
            # Store as potential fallback
            if n < best_n:
                best_n = n
                best_result = (n, edges)
    
    # Try GP graphs
    if _time_left() > 20.0:
        # For GP graphs with 2n vertices, try to find ones with girth >= g
        # The girth of GP(n,k) is related to the order of k in Z_n
        # and the structure of the graph
        for n in range(300, 600):
            if _time_left() < 10.0:
                break
            for k_gp in range(2, n):
                if _time_left() < 5.0:
                    break
                if math.gcd(n, k_gp) != 1:
                    continue
                # Quick girth estimate: need no short cycles
                # A cycle in GP(n,k) corresponds to a relation like:
                # a*1 + b*k = 0 mod n for some small a,b
                # Check for short relations
                valid = True
                for a in range(1, g):
                    for b in range(1, g):
                        if (a + b * k_gp) % n == 0:
                            if a + b < g:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Construct and verify
                total_n = 2 * n
                adj_gp = [[] for _ in range(total_n)]
                edges_gp = []
                for i in range(n):
                    u1, v1 = i, n + i
                    u2, v2 = (i + 1) % n, (i + k_gp) % n
                    edges_gp.append((u1, v1))
                    adj_gp[u1].append(v1)
                    adj_gp[v1].append(u1)
                    edges_gp.append((u1, u2))
                    adj_gp[u1].append(u2)
                    adj_gp[u2].append(u1)
                    edges_gp.append((v1, n + v2))
                    adj_gp[v1].append(n + v2)
                    adj_gp[n + v2].append(v1)
                
                if _check_girth_fast(total_n, adj_gp, g):
                    if total_n < best_n:
                        best_n = total_n
                        best_result = (total_n, edges_gp)
                    return total_n, edges_gp
    
    # Return best result found
    if best_result is not None:
        return best_result
    
    # Last resort: return a LUW graph even if not verified
    result = _construct_luw(q, min_d)
    if result is not None:
        return result
    
    # Absolute last resort: cycle
    n = max(g, 6)
    edges = [(i, (i + 1) % n) for i in range(n)]
    return n, edges
=======
def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    best_result = None
    best_n = float('inf')
    
    q = k
    
    # Try LUW construction with varying d
    # Girth of LUW(d, q) is approx 2d+2
    # We need girth >= g
    # So 2d+2 >= g => d >= (g-2)/2
    # For g=16, d >= 7
    # But LUW(d=7, q=3) has ~2916 vertices, too many
    # LUW(d=6, q=3) has ~972 vertices but girth only 14
    # Let's try d=6 first (might have higher actual girth), then d=7
    
    min_d = max(3, (g - 2) // 2)
    
    # Try d from min_d-1 to min_d+3 (allow some flexibility)
    for d in range(max(3, min_d - 1), min_d + 4):
        if _time_left() < 10.0:
            break
            
        # Construct LUW graph
        result = _construct_luw(q, d)
        if result is None:
            continue
            
        n, edges = result
        
        # Verify girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if _check_girth_fast(n, adj, g):
            if n < best_n:
                best_n = n
                best_result = (n, edges)
            return n, edges
        else:
            # Store as potential fallback
            if n < best_n:
                best_n = n
                best_result = (n, edges)
    
    # Try GP graphs with fast algebraic pre-filter
    if _time_left() > 20.0:
        # For GP graphs with 2n vertices, try to find ones with girth >= g
        # Use algebraic pre-filter to avoid expensive BFS
        # Search n from 468 (giving 936 vertices) upward
        for n in range(468, 700):
            if _time_left() < 10.0:
                break
            for k_gp in range(2, n):
                if _time_left() < 5.0:
                    break
                if math.gcd(n, k_gp) != 1:
                    continue
                # Fast algebraic girth check
                est_girth = _gp_girth_fast(n, k_gp)
                if est_girth < g:
                    continue
                
                # Construct and verify with full BFS
                total_n = 2 * n
                adj_gp = [[] for _ in range(total_n)]
                edges_gp = []
                for i in range(n):
                    u1, v1 = i, n + i
                    u2, v2 = (i + 1) % n, (i + k_gp) % n
                    edges_gp.append((u1, v1))
                    adj_gp[u1].append(v1)
                    adj_gp[v1].append(u1)
                    edges_gp.append((u1, u2))
                    adj_gp[u1].append(u2)
                    adj_gp[u2].append(u1)
                    edges_gp.append((v1, n + v2))
                    adj_gp[v1].append(n + v2)
                    adj_gp[n + v2].append(v1)
                
                if _check_girth_fast(total_n, adj_gp, g):
                    if total_n < best_n:
                        best_n = total_n
                        best_result = (total_n, edges_gp)
                    return total_n, edges_gp
    
    # Return best result found
    if best_result is not None:
        return best_result
    
    # Last resort: return a LUW graph even if not verified
    result = _construct_luw(q, min_d)
    if result is not None:
        return result
    
    # Absolute last resort: cycle
    n = max(g, 6)
    edges = [(i, (i + 1) % n) for i in range(n)]
    return n, edges
>>>>>>> REPLACE

