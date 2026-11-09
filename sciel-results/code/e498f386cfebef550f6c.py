import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    # Helper: Check if graph is C4-free
    def is_c4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common.bit_count() > 1:
                    return False
        return True

    # Helper: Check if alpha < 11 (i.e., no independent set of size 11)
    def has_independent_set_11(adj, n):
        full_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            non_neighbors[i] = full_mask ^ adj[i] ^ (1 << i)
            
        def backtrack(candidates, size):
            if size == 11:
                return True
            if len(candidates) < 11 - size:
                return False
            
            v = candidates[0]
            rest = candidates[1:]
            
            new_candidates = [u for u in rest if (non_neighbors[v] >> u) & 1]
            if backtrack(new_candidates, size + 1):
                return True
            
            if backtrack(rest, size):
                return True
                
            return False
            
        return backtrack(list(range(n)), 0)

    # Fallback: Seed program (n=30)
    def get_seed_graph():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # Polarity graph of PG(2, q^2) - C4-free by construction
    # PG(2, 5) has q=5, so 5^2 = 25 points + 1 = 26? 
    # Actually PG(2, q) has (q^2 + q + 1) points
    # For q=5: 25+5+1 = 31 points
    # The polarity graph connects point P to line P^perp
    # This is C4-free and has alpha bounded by q+1 = 6
    def get_polarity_graph_q5():
        q = 5
        # Points in PG(2, q): represented as (x, y, 1) for affine, plus points at infinity
        # Total: q^2 + q + 1 = 25 + 5 + 1 = 31 points
        
        # Generate all points as triples (x, y, z) with z != 0 normalized, plus points at infinity
        points = []
        # Affine points: (x, y, 1) for x, y in GF(q)
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        # Points at infinity: (1, m, 0) for m in GF(q), and (0, 1, 0)
        for m in range(q):
            points.append((1, m, 0))
        points.append((0, 1, 0))
        
        n = len(points)  # Should be 31
        
        # Polarity: (x, y, z) is orthogonal to (x', y', z') if x*x' + y*y' + z*z' = 0 in GF(q)
        # For q=5, we work mod 5
        
        adj = [0] * n
        edges = []
        
        for i in range(n):
            xi, yi, zi = points[i]
            for j in range(i + 1, n):
                xj, yj, zj = points[j]
                # Check if xi*xj + yi*yj + zi*zj ≡ 0 (mod 5)
                if (xi * xj + yi * yj + zi * zj) % q == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        
        # Verify C4-free
        assert is_c4_free(adj, n), "Polarity graph should be C4-free"
        
        # Check independence number
        if not has_independent_set_11(adj, n):
            return {'n': n, 'edges': edges}
        
        return None

    # Try polarity graph first
    result = get_polarity_graph_q5()
    if result is not None:
        return result
    
    # Try to construct for n from 39 down to 31
    for n in range(39, 30, -1):
        if time.time() > deadline:
            break
            
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Generate random graph
            p = 0.3 + 0.2 * random.random()
            adj = [0] * n
            edges = []
            
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
                        
            # Remove C4s
            for _ in range(100):
                # Find a C4
                c4_edge = None
                for u in range(n):
                    for v in range(u + 1, n):
                        common = adj[u] & adj[v]
                        if common.bit_count() > 1:
                            bits = []
                            mask = common
                            while mask:
                                lsb = mask & -mask
                                bits.append(lsb.bit_length() - 1)
                                mask ^= lsb
                                if len(bits) == 2:
                                    break
                            c4_edge = (u, bits[0])
                            break
                    if c4_edge:
                        break
                if c4_edge is None:
                    break
                u, v = c4_edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            
            # Rebuild edges list
            edges = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    mask ^= lsb
            
            if not is_c4_free(adj, n):
                continue
                
            if not has_independent_set_11(adj, n):
                return {'n': n, 'edges': edges}

    # If no graph found for n > 30, return seed
    return get_seed_graph()
