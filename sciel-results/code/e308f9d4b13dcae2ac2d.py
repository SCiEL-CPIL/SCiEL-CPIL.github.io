import time
import random

def construct():
    deadline = time.time() + 115.0
    
    # Use the polarity graph of PG(2,6)
    # PG(2, q) has q^2 + q + 1 points
    # For q=6, that's 36 + 6 + 1 = 43 vertices
    # The polarity graph has edges between orthogonal points
    # It is C4-free and has small independence number
    
    q = 6
    
    # Generate all non-zero vectors in F_q^3, normalized
    # F_6 is not a field (6 is not prime), so we need a different approach
    # Actually, PG(2,6) doesn't exist as a projective plane of prime power order
    # We need to use a different construction
    
    # Let's use the known construction: the polarity graph of the Hermitian unital
    # Or better: use the Gewirtz graph or similar strongly regular graph
    
    # Actually, let me use a known C4-free graph with n=39 (the best known)
    # The polarity graph of PG(2, 6) doesn't work since 6 is not a prime power
    
    # Let's use q=5 (PG(2,5) polarity graph, n=31) and try to extend it
    # Or use q=7 (PG(2,7) polarity graph, n=57) and take a subgraph
    
    # Better approach: use the known (39, 8) C4-free graph
    # This is the polarity graph of the symplectic polar space W(5, 2)
    # Or the complement of the Clebsch graph extended...
    
    # Let me construct the polarity graph of PG(2, 5) first (n=31, C4-free)
    # Then try to add vertices greedily with C4 constraint
    
    # Actually, the best known construction for R(C4, K11) > 39 uses:
    # The polarity graph of the Hermitian variety H(2, 32) or similar
    # Or the (39, 8, 1, 1) strongly regular graph
    
    # Let me use a concrete construction: the Gewirtz graph is (56, 10, 0, 2) - no C4
    # But I need n up to 43 with no independent 11-set
    
    # Strategy: 
    # 1. Build PG(2,5) polarity graph (n=31, C4-free, alpha ~ 5-6)
    # 2. Try to add vertices one by one, maintaining C4-free and alpha < 11
    # 3. Target n=39 or higher
    
    def build_polarity_graph_q5():
        """Build the polarity graph of PG(2,5). n=31, C4-free."""
        q = 5
        points = []
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv = pow(x, q-2, q)
                        points.append((1, (y * inv) % q, (z * inv) % q))
                    elif y != 0:
                        inv = pow(y, q-2, q)
                        points.append((0, 1, (z * inv) % q))
                    else:
                        points.append((0, 0, 1))
        
        n_base = len(points)  # Should be 31
        adj = [0] * n_base
        for i in range(n_base):
            for j in range(i + 1, n_base):
                u = points[i]
                v = points[j]
                dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
                if dot == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n_base
    
    def is_c4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common & (common - 1):  # more than one bit set
                    return False
        return True
    
    def alpha_less_than(adj, n, limit):
        """Check if independence number < limit using branch and bound."""
        # Greedy lower bound first
        remaining = (1 << n) - 1
        count = 0
        while remaining:
            best_v = -1
            best_deg = n + 1
            for v in range(n):
                if remaining & (1 << v):
                    non_adj = remaining & ~(adj[v] | (1 << v))
                    deg = bin(non_adj).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            count += 1
            non_adj = remaining & ~(adj[best_v] | (1 << best_v))
            remaining &= ~(non_adj | (1 << best_v))
            if count >= limit:
                return False  # Found independent set of size >= limit
        
        # More careful branch and bound
        def backtrack(candidates, size):
            if size >= limit:
                return True  # Found independent set of size >= limit
            if candidates == 0:
                return False
            if time.time() > deadline - 15:
                return False
            
            # Upper bound
            if size + bin(candidates).count('1') < limit:
                return False
            
            # Pick vertex with minimum degree in candidates
            best_v = -1
            best_deg = n + 1
            temp = candidates
            while temp:
                v = temp & (-temp)
                v_idx = v.bit_length() - 1
                temp ^= v
                non_adj = candidates & ~(adj[v_idx] | (1 << v_idx))
                deg = bin(non_adj).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v_idx
            
            if best_v == -1:
                return False
            
            v_bit = 1 << best_v
            
            # Option 1: Don't include best_v
            if backtrack(candidates & ~v_bit, size):
                return True
            
            # Option 2: Include best_v, remove it and its neighbors
            new_candidates = candidates & ~(adj[best_v] | v_bit)
            if backtrack(new_candidates, size + 1):
                return True
            
            return False
        
        return not backtrack((1 << n) - 1, 0)
    
    # Build base graph
    base_adj, base_n = build_polarity_graph_q5()
    
    # Verify base graph
    if not is_c4_free(base_adj, base_n):
        # Fallback: disjoint triangles
        n_fallback = 30
        adj_fallback = [0] * n_fallback
        for i in range(10):
            u, v, w = 3*i, 3*i+1, 3*i+2
            adj_fallback[u] |= (1<<v) | (1<<w)
            adj_fallback[v] |= (1<<u) | (1<<w)
            adj_fallback[w] |= (1<<u) | (1<<v)
        edges = []
        for u in range(n_fallback):
            for v in range(u + 1, n_fallback):
                if adj_fallback[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n_fallback, 'edges': edges}
    
    # Verify alpha < 11 for base
    if not alpha_less_than(base_adj, base_n, 11):
        # Base graph has independent set of size 11, try different approach
        # For now, still try to extend it
        pass
    
    # Try to add vertices
    current_adj = base_adj[:]
    current_n = base_n
    
    random.seed(12345)
    
    # Precompute pair common neighbor info for efficient C4 checking
    def can_add_vertex(adj_list, n, new_adj_mask):
        """Check if adding a vertex with adjacency new_adj_mask maintains C4-freeness."""
        # For any pair u,v both adjacent to new vertex, they must have no existing common neighbor
        # (since the new vertex would be their 2nd common neighbor)
        for u in range(n):
            if not (new_adj_mask & (1 << u)):
                continue
            for v in range(u + 1, n):
                if not (new_adj_mask & (1 << v)):
                    continue
                # u and v are both adjacent to new vertex
                # Check if they already share a common neighbor
                common = adj_list[u] & adj_list[v]
                if common != 0:
                    return False
        return True
    
    # Try to add up to 12 vertices (target n=43)
    for attempt_num in range(12):
        if time.time() > deadline - 30:
            break
        
        found = False
        # Try multiple random patterns
        for trial in range(200):
            if time.time() > deadline - 30:
                break
            
            # Generate random adjacency pattern
            # Bias towards connecting to more vertices (to keep alpha small)
            new_adj_mask = 0
            for v in range(current_n):
                if random.random() < 0.65:
                    new_adj_mask |= (1 << v)
            
            # Check C4-freeness
            if not can_add_vertex(current_adj, current_n, new_adj_mask):
                continue
            
            # Build new adjacency
            new_adj = current_adj[:] + [0]
            for v in range(current_n):
                if new_adj_mask & (1 << v):
                    new_adj[current_n] |= (1 << v)
                    new_adj[v] |= (1 << current_n)
            
            # Check alpha < 11
            if not alpha_less_than(new_adj, current_n + 1, 11):
                continue
            
            # Success! Update
            current_adj = new_adj
            current_n += 1
            found = True
            break
        
        if not found:
            break
    
    # Final validation
    if not is_c4_free(current_adj, current_n):
        n_fallback = 30
        adj_fallback = [0] * n_fallback
        for i in range(10):
            u, v, w = 3*i, 3*i+1, 3*i+2
            adj_fallback[u] |= (1<<v) | (1<<w)
            adj_fallback[v] |= (1<<u) | (1<<w)
            adj_fallback[w] |= (1<<u) | (1<<v)
        edges = []
        for u in range(n_fallback):
            for v in range(u + 1, n_fallback):
                if adj_fallback[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n_fallback, 'edges': edges}
    
    if not alpha_less_than(current_adj, current_n, 11):
        # Fall back to base if it was valid
        if is_c4_free(base_adj, base_n) and alpha_less_than(base_adj, base_n, 11):
            current_adj = base_adj
            current_n = base_n
        else:
            n_fallback = 30
            adj_fallback = [0] * n_fallback
            for i in range(10):
                u, v, w = 3*i, 3*i+1, 3*i+2
                adj_fallback[u] |= (1<<v) | (1<<w)
                adj_fallback[v] |= (1<<u) | (1<<w)
                adj_fallback[w] |= (1<<u) | (1<<v)
            edges = []
            for u in range(n_fallback):
                for v in range(u + 1, n_fallback):
                    if adj_fallback[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n_fallback, 'edges': edges}
    
    # Build edge list
    edges = []
    for u in range(current_n):
        for v in range(u + 1, current_n):
            if current_adj[u] & (1 << v):
                edges.append([u, v])
    
    return {'n': current_n, 'edges': edges}
