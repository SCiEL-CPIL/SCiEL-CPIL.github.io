import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Build the 17-vertex Paley graph as fallback
    squares17 = {x * x % 17 for x in range(1, 17)}
    adj17 = np.zeros((17, 17), dtype=np.int8)
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares17:
                adj17[i, j] = 1
                adj17[j, i] = 1
    
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if adj17[i, j]]
    best_n = 17
    best_edges = edges_17
    
    def has_k4_bitmask(n, masks):
        """Check if graph has any K4 using bitmasks."""
        for u in range(n):
            for v in range(u + 1, n):
                if masks[u] & (1 << v):
                    common = masks[u] & masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if masks[wi] & (1 << xi):
                                return True
        return False
    
    def has_ind6_bitmask(n, comp_masks):
        """Check if graph has any independent set of size 6 using bitmasks."""
        for u in range(n):
            for v in range(u + 1, n):
                if comp_masks[u] & (1 << v):
                    common = comp_masks[u] & comp_masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if comp_masks[wi] & (1 << xi):
                                common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                                common2 &= ~((1 << (xi + 1)) - 1)
                                if common2:
                                    return True
        return False
    
    def build_masks_from_adj(n, adj):
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks
    
    def build_comp_masks_from_masks(n, masks):
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        return comp_masks
    
    def verify_and_get_edges(n, adj):
        """Verify graph is K4-free and has no independent 6-set, return edges if valid."""
        masks = build_masks_from_adj(n, adj)
        comp_masks = build_comp_masks_from_masks(n, masks)
        if has_k4_bitmask(n, masks) or has_ind6_bitmask(n, comp_masks):
            return None
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
        return edges
    
    # === ALGEBRAIC CONSTRUCTION USING FINITE FIELD GEOMETRY ===
    # Strategy: Use points in GF(q)^2 or related structures with carefully chosen
    # edge definitions to create K4-free graphs with small independence numbers.
    
    def try_finite_field_construction(q, time_limit):
        """
        Try to construct a graph using finite field GF(q) geometry.
        Vertices are points in some affine/projective space over GF(q).
        Edge definition based on quadratic forms or multiplicative characters.
        """
        if time.time() > start_time + time_limit:
            return None
        
        # For GF(q) where q is prime or prime power
        # We'll use different constructions based on q
        
        if q == 13:
            # Paley-type: vertices 0..12, edge if (j-i) mod 13 is a quadratic residue
            squares = {x * x % q for x in range(1, q)}
            adj = np.zeros((q, q), dtype=np.int8)
            for i in range(q):
                for j in range(i + 1, q):
                    if (j - i) % q in squares:
                        adj[i, j] = 1
                        adj[j, i] = 1
            edges = verify_and_get_edges(q, adj)
            if edges is not None:
                return {'n': q, 'edges': edges}
        
        elif q == 16:
            # GF(16) construction: use multiplicative group properties
            # Represent GF(16) elements as pairs (a,b) in GF(2)^2
            # Edge if the "difference" has specific property
            
            # GF(16) arithmetic: x^4 + x + 1 = 0
            # Element representation: 4-bit integer
            # Multiplication table for GF(16)
            def gf16_mult(a, b):
                if a == 0 or b == 0:
                    return 0
                result = 0
                while b:
                    if b & 1:
                        result ^= a
                    a <<= 1
                    if a & 16:
                        a ^= 17  # x^4 + x + 1
                    b >>= 1
                return result
            
            def gf16_add(a, b):
                return a ^ b
            
            # Trace function: Tr(x) = x + x^2 + x^4 + x^8
            def gf16_trace(x):
                return gf16_add(gf16_add(gf16_add(x, gf16_mult(x, x)), gf16_mult(gf16_mult(x, x), gf16_mult(gf16_mult(x, x)))), gf16_mult(gf16_mult(gf16_mult(gf16_mult(x, x), x), x), gf16_mult(gf16_mult(gf16_mult(gf16_mult(x, x), x), x))))
            
            # Alternative: use the fact that GF(16) has a cyclic multiplicative group of order 15
            # Generator: 3 (since 3^15 = 1 in GF(16))
            # Let's verify: 3^1=3, 3^2=9, 3^3=27=10 (27^16=11, 27^16^2=11^16=1, so 27^16=11, 27^16^2=1)
            # Actually let me just use a simpler approach
            
            # Use Paley graph on 16 vertices? No, 16 is not prime.
            # Instead, try a random graph with targeted density and verify
            
            # Better: use the known (16, 56, 4) design or similar
            # For now, skip q=16 and try other approaches
        
        elif q == 25:
            # GF(25) construction
            # x^2 + x + 2 = 0 over GF(5) is irreducible
            # Elements: (a, b) where a,b in GF(5), representing a + b*alpha
            # alpha^2 = -alpha - 2 = 3alpha + 3 (mod 5)
            
            def gf25_add(a, b):
                return ((a[0] + b[0]) % 5, (a[1] + b[1]) % 5)
            
            def gf25_neg(a):
                return ((-a[0]) % 5, (-a[1]) % 5)
            
            def gf25_mult(a, b):
                a0, a1 = a
                b0, b1 = b
                # (a0 + a1*alpha)(b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
                # alpha^2 = 3*alpha + 3
                # = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*(3*alpha + 3)
                # = (a0*b0 + 3*a1*b1) + (a0*b1 + a1*b0 + 3*a1*b1)*alpha
                result0 = (a0 * b0 + 3 * a1 * b1) % 5
                result1 = (a0 * b1 + a1 * b0 + 3 * a1 * b1) % 5
                return (result0, result1)
            
            # Quadratic character: x is a square if x^12 = 1 (for x != 0)
            # Or: compute x^(12) and check if it equals 1
            def gf25_pow(x, n):
                result = (1, 0)
                base = x
                while n > 0:
                    if n & 1:
                        result = gf25_mult(result, base)
                    base = gf25_mult(base, base)
                    n >>= 1
                return result
            
            def gf25_is_square(x):
                if x == (0, 0):
                    return False
                x12 = gf25_pow(x, 12)
                return x12 == (1, 0)
            
            # Construct: vertices are the 25 elements of GF(25)
            # Edge (i,j) if j-i is a square in GF(25)
            elements = [(a, b) for a in range(5) for b in range(5)]
            elem_to_idx = {e: i for i, e in enumerate(elements)}
            
            adj = np.zeros((25, 25), dtype=np.int8)
            for i in range(25):
                for j in range(i + 1, 25):
                    diff = gf25_add(elements[j], gf25_neg(elements[i]))
                    if gf25_is_square(diff):
                        adj[i, j] = 1
                        adj[j, i] = 1
            
            edges = verify_and_get_edges(25, adj)
            if edges is not None:
                return {'n': 25, 'edges': edges}
        
        elif q == 49:
            # GF(49) = GF(7^2)
            # x^2 + 3x + 5 = 0 over GF(7) is irreducible (discriminant 9-20=-11=3, not a square mod 7)
            # Squares mod 7: 1, 2, 4. 3 is not a square, so irreducible.
            
            def gf49_add(a, b):
                return ((a[0] + b[0]) % 7, (a[1] + b[1]) % 7)
            
            def gf49_neg(a):
                return ((-a[0]) % 7, (-a[1]) % 7)
            
            def gf49_mult(a, b):
                a0, a1 = a
                b0, b1 = b
                # alpha^2 = -3*alpha - 5 = 4*alpha + 2 (mod 7)
                result0 = (a0 * b0 + 2 * a1 * b1) % 7
                result1 = (a0 * b1 + a1 * b0 + 4 * a1 * b1) % 7
                return (result0, result1)
            
            def gf49_pow(x, n):
                result = (1, 0)
                base = x
                while n > 0:
                    if n & 1:
                        result = gf49_mult(result, base)
                    base = gf49_mult(base, base)
                    n >>= 1
                return result
            
            def gf49_is_square(x):
                if x == (0, 0):
                    return False
                x24 = gf49_pow(x, 24)
                return x24 == (1, 0)
            
            elements = [(a, b) for a in range(7) for b in range(7)]
            
            adj = np.zeros((49, 49), dtype=np.int8)
            for i in range(49):
                for j in range(i + 1, 49):
                    diff = gf49_add(elements[j], gf49_neg(elements[i]))
                    if gf49_is_square(diff):
                        adj[i, j] = 1
                        adj[j, i] = 1
            
            edges = verify_and_get_edges(49, adj)
            if edges is not None:
                return {'n': 49, 'edges': edges}
        
        return None
    
    # Try algebraic constructions in order of increasing n
    for q in [13, 25, 49]:
        if time.time() > deadline - 10.0:
            break
        result = try_finite_field_construction(q, deadline - time.time() - 5.0)
        if result is not None:
            if result['n'] > best_n:
                best_n = result['n']
                best_edges = result['edges']
            # Continue trying larger constructions
    
    # If we found n >= 35, we're done (can't exceed 40, and 35 is the record)
    if best_n >= 35:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: try local search from the best algebraic result or Paley(17)
    if time.time() > deadline - 5.0:
        return {'n': best_n, 'edges': best_edges}
    
    # Use local search to try to extend from current best
    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks
    
    def build_comp_masks(n, masks):
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        return comp_masks
    
    def count_violations(n, masks, comp_masks):
        """Count K4s and independent 6-sets."""
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if masks[u] & (1 << v):
                    common = masks[u] & masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if masks[wi] & (1 << xi):
                                k4_count += 1
        
        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if comp_masks[u] & (1 << v):
                    common = comp_masks[u] & comp_masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if comp_masks[wi] & (1 << xi):
                                common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                                common2 &= ~((1 << (xi + 1)) - 1)
                                if common2:
                                    ind6_count += 1
        return k4_count + ind6_count
    
    def try_repair(n, time_budget, seed_adj=None):
        """Local search with targeted repair strategy."""
        end_time = time.time() + time_budget
        
        if seed_adj is not None and seed_adj.shape == (n, n):
            adj = seed_adj.copy()
        else:
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
        
        masks = build_masks(n, adj)
        comp_masks = build_comp_masks(n, masks)
        current_cost = count_violations(n, masks, comp_masks)
        
        if current_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges
        
        T = 2.0
        min_T = 0.05
        cooling = 0.9999
        no_improve = 0
        max_no_improve = 5000
        best_cost = current_cost
        best_adj = adj.copy()
        
        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            old_state = adj[u, v]
            
            if old_state == 0:
                common_nv = masks[u] & masks[v]
                common_nv &= ~(1 << u)
                common_nv &= ~(1 << v)
                new_k4 = 0
                w_mask = common_nv
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common_nv & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            new_k4 += 1
                
                destroyed_ind6 = 0
                if comp_masks[u] & (1 << v):
                    common_cv = comp_masks[u] & comp_masks[v]
                    common_cv &= ~((1 << (v + 1)) - 1)
                    w_mask2 = common_cv
                    while w_mask2:
                        lsb_w2 = w_mask2 & -w_mask2
                        wi2 = lsb_w2.bit_length() - 1
                        w_mask2 ^= lsb_w2
                        x_mask2 = common_cv & ~((1 << (wi2 + 1)) - 1)
                        while x_mask2:
                            lsb_x2 = x_mask2 & -x_mask2
                            xi2 = lsb_x2.bit_length() - 1
                            x_mask2 ^= lsb_x2
                            if comp_masks[wi2] & (1 << xi2):
                                common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi2] & comp_masks[xi2]
                                common2 &= ~((1 << (xi2 + 1)) - 1)
                                if common2:
                                    destroyed_ind6 += 1
                
                delta = new_k4 - destroyed_ind6
            else:
                destroyed_k4 = 0
                if masks[u] & (1 << v):
                    common_mv = masks[u] & masks[v]
                    common_mv &= ~((1 << (v + 1)) - 1)
                    w_mask3 = common_mv
                    while w_mask3:
                        lsb_w3 = w_mask3 & -w_mask3
                        wi3 = lsb_w3.bit_length() - 1
                        w_mask3 ^= lsb_w3
                        x_mask3 = common_mv & ~((1 << (wi3 + 1)) - 1)
                        while x_mask3:
                            lsb_x3 = x_mask3 & -x_mask3
                            xi3 = lsb_x3.bit_length() - 1
                            x_mask3 ^= lsb_x3
                            if masks[wi3] & (1 << xi3):
                                destroyed_k4 += 1
                
                new_ind6 = 0
                temp_comp_u = comp_masks[u] | (1 << v)
                temp_comp_v = comp_masks[v] | (1 << u)
                common_cv2 = temp_comp_u & temp_comp_v
                common_cv2 &= ~((1 << (v + 1)) - 1)
                w_mask4 = common_cv2
                while w_mask4:
                    lsb_w4 = w_mask4 & -w_mask4
                    wi4 = lsb_w4.bit_length() - 1
                    w_mask4 ^= lsb_w4
                    x_mask4 = common_cv2 & ~((1 << (wi4 + 1)) - 1)
                    while x_mask4:
                        lsb_x4 = x_mask4 & -x_mask4
                        xi4 = lsb_x4.bit_length() - 1
                        x_mask4 ^= lsb_x4
                        if (comp_masks[wi4] & (1 << xi4)) and (comp_masks[wi4] & (1 << u)) and (comp_masks[wi4] & (1 << v)) and (comp_masks[xi4] & (1 << u)) and (comp_masks[xi4] & (1 << v)):
                            new_ind6 += 1
                
                delta = -destroyed_k4 + new_ind6
            
            if delta < 0:
                adj[u, v] = 1
                adj[v, u] = 1
                if old_state == 0:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    comp_masks[u] &= ~(1 << v)
                    comp_masks[v] &= ~(1 << u)
                else:
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    comp_masks[u] |= (1 << v)
                    comp_masks[v] |= (1 << u)
                current_cost += delta
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj.copy()
                if current_cost <= 0:
                    if not has_k4_bitmask(n, masks) and not has_ind6_bitmask(n, comp_masks):
                        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                        return edges
            else:
                T = max(min_T, T * cooling)
                if random.random() < np.exp(-delta / T):
                    adj[u, v] = 1
                    adj[v, u] = 1
                    if old_state == 0:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        comp_masks[u] &= ~(1 << v)
                        comp_masks[v] &= ~(1 << u)
                    else:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                        comp_masks[u] |= (1 << v)
                        comp_masks[v] |= (1 << u)
                    current_cost += delta
                    no_improve += 1
                else:
                    no_improve += 1
        
        if not has_k4_bitmask(n, masks) and not has_ind6_bitmask(n, comp_masks):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges
        if best_cost < current_cost:
            best_masks = build_masks(n, best_adj)
            best_comp = build_comp_masks(n, best_masks)
            if not has_k4_bitmask(n, best_masks) and not has_ind6_bitmask(n, best_comp):
                edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
                return edges
        return None
    
    # Try to extend from best_n to 35 using local search
    current_n = best_n
    # Reconstruct adj from best_edges
    current_adj = np.zeros((current_n, current_n), dtype=np.int8)
    for u, v in best_edges:
        current_adj[u, v] = 1
        current_adj[v, u] = 1
    
    while current_n < 35 and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        if remaining < 1.0:
            break
        
        target_n = current_n + 1
        budget = min(remaining * 0.25, 8.0)
        
        found = None
        num_attempts = 3
        per_attempt = budget / num_attempts
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5.0:
                break
            
            seed_adj = np.zeros((target_n, target_n), dtype=np.int8)
            seed_adj[:current_n, :current_n] = current_adj
            
            p_connect = 0.4 + 0.1 * random.random()
            for j in range(current_n):
                if random.random() < p_connect:
                    seed_adj[current_n, j] = 1
                    seed_adj[j, current_n] = 1
            
            found = try_repair(target_n, per_attempt, seed_adj=seed_adj)
            if found is not None:
                break
        
        if found is not None:
            current_n = target_n
            current_adj = np.zeros((current_n, current_n), dtype=np.int8)
            for u, v in found:
                current_adj[u, v] = 1
                current_adj[v, u] = 1
            best_n = current_n
            best_edges = found
        else:
            if current_n >= 25:
                break
    
    # Final attempt
    if best_n < 35 and time.time() < deadline - 3.0:
        remaining = deadline - time.time() - 3.0
        for target in range(max(best_n + 1, 30), 36):
            if time.time() > deadline - 2.0:
                break
            budget = min(remaining * 0.15, 6.0)
            found = try_repair(target, budget)
            if found is not None:
                best_n = target
                best_edges = found
                break
    
    return {'n': best_n, 'edges': best_edges}
    
    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks
    
    def build_comp_masks(n, masks):
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        return comp_masks
    
    def has_k4(n, masks):
        """Check if graph has any K4 using bitmasks."""
        for u in range(n):
            for v in range(u + 1, n):
                if masks[u] & (1 << v):
                    common = masks[u] & masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if masks[wi] & (1 << xi):
                                return True
        return False
    
    def has_ind6(n, comp_masks):
        """Check if graph has any independent set of size 6 using bitmasks."""
        for u in range(n):
            for v in range(u + 1, n):
                if comp_masks[u] & (1 << v):
                    common = comp_masks[u] & comp_masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if comp_masks[wi] & (1 << xi):
                                common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                                common2 &= ~((1 << (xi + 1)) - 1)
                                if common2:
                                    return True
        return False
    
    def is_valid(n, masks, comp_masks):
        return not has_k4(n, masks) and not has_ind6(n, comp_masks)
    
    def count_violations(n, masks, comp_masks):
        """Count K4s and independent 6-sets."""
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if masks[u] & (1 << v):
                    common = masks[u] & masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if masks[wi] & (1 << xi):
                                k4_count += 1
        
        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if comp_masks[u] & (1 << v):
                    common = comp_masks[u] & comp_masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if comp_masks[wi] & (1 << xi):
                                common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                                common2 &= ~((1 << (xi + 1)) - 1)
                                if common2:
                                    ind6_count += 1
        return k4_count + ind6_count
    
    def try_repair(n, time_budget, seed_adj=None):
        """Local search with targeted repair strategy."""
        end_time = time.time() + time_budget
        
        if seed_adj is not None and seed_adj.shape == (n, n):
            adj = seed_adj.copy()
        else:
            # Start from Paley(17) embedded in first 17 vertices, random rest
            adj = np.zeros((n, n), dtype=np.int8)
            adj[:17, :17] = adj17
            for i in range(17, n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
            for i in range(17, n):
                for j in range(17):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
        
        masks = build_masks(n, adj)
        comp_masks = build_comp_masks(n, masks)
        current_cost = count_violations(n, masks, comp_masks)
        
        if current_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges
        
        # Targeted repair: prioritize moves that fix violations
        T = 2.0
        min_T = 0.05
        cooling = 0.9999
        no_improve = 0
        max_no_improve = 5000
        full_mask = (1 << n) - 1
        best_cost = current_cost
        best_adj = adj.copy()
        
        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            old_state = adj[u, v]
            
            if old_state == 0:
                # Adding edge (u,v)
                common_nv = masks[u] & masks[v]
                common_nv &= ~(1 << u)
                common_nv &= ~(1 << v)
                new_k4 = 0
                w_mask = common_nv
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common_nv & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            new_k4 += 1
                
                # Count ind6 destroyed (u,v become adjacent)
                destroyed_ind6 = 0
                if comp_masks[u] & (1 << v):
                    common_cv = comp_masks[u] & comp_masks[v]
                    common_cv &= ~((1 << (v + 1)) - 1)
                    w_mask2 = common_cv
                    while w_mask2:
                        lsb_w2 = w_mask2 & -w_mask2
                        wi2 = lsb_w2.bit_length() - 1
                        w_mask2 ^= lsb_w2
                        x_mask2 = common_cv & ~((1 << (wi2 + 1)) - 1)
                        while x_mask2:
                            lsb_x2 = x_mask2 & -x_mask2
                            xi2 = lsb_x2.bit_length() - 1
                            x_mask2 ^= lsb_x2
                            if comp_masks[wi2] & (1 << xi2):
                                common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi2] & comp_masks[xi2]
                                common2 &= ~((1 << (xi2 + 1)) - 1)
                                if common2:
                                    destroyed_ind6 += 1
                
                delta = new_k4 - destroyed_ind6
            else:
                # Removing edge (u,v)
                destroyed_k4 = 0
                if masks[u] & (1 << v):
                    common_mv = masks[u] & masks[v]
                    common_mv &= ~((1 << (v + 1)) - 1)
                    w_mask3 = common_mv
                    while w_mask3:
                        lsb_w3 = w_mask3 & -w_mask3
                        wi3 = lsb_w3.bit_length() - 1
                        w_mask3 ^= lsb_w3
                        x_mask3 = common_mv & ~((1 << (wi3 + 1)) - 1)
                        while x_mask3:
                            lsb_x3 = x_mask3 & -x_mask3
                            xi3 = lsb_x3.bit_length() - 1
                            x_mask3 ^= lsb_x3
                            if masks[wi3] & (1 << xi3):
                                destroyed_k4 += 1
                
                # New ind6 created (u,v become non-adjacent)
                new_ind6 = 0
                # Temporarily mark u,v as non-adjacent
                temp_comp_u = comp_masks[u] | (1 << v)
                temp_comp_v = comp_masks[v] | (1 << u)
                common_cv2 = temp_comp_u & temp_comp_v
                common_cv2 &= ~((1 << (v + 1)) - 1)
                w_mask4 = common_cv2
                while w_mask4:
                    lsb_w4 = w_mask4 & -w_mask4
                    wi4 = lsb_w4.bit_length() - 1
                    w_mask4 ^= lsb_w4
                    x_mask4 = common_cv2 & ~((1 << (wi4 + 1)) - 1)
                    while x_mask4:
                        lsb_x4 = x_mask4 & -x_mask4
                        xi4 = lsb_x4.bit_length() - 1
                        x_mask4 ^= lsb_x4
                        # Check if wi4 and xi4 are both non-adjacent to each other and to u,v
                        if (comp_masks[wi4] & (1 << xi4)) and (comp_masks[wi4] & (1 << u)) and (comp_masks[wi4] & (1 << v)) and (comp_masks[xi4] & (1 << u)) and (comp_masks[xi4] & (1 << v)):
                            new_ind6 += 1
                
                delta = -destroyed_k4 + new_ind6
            
            if delta < 0:
                # Accept improving move
                adj[u, v] = 1
                adj[v, u] = 1
                if old_state == 0:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    comp_masks[u] &= ~(1 << v)
                    comp_masks[v] &= ~(1 << u)
                else:
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    comp_masks[u] |= (1 << v)
                    comp_masks[v] |= (1 << u)
                current_cost += delta
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj.copy()
                if current_cost <= 0:
                    if is_valid(n, masks, comp_masks):
                        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                        return edges
            else:
                T = max(min_T, T * cooling)
                if random.random() < np.exp(-delta / T):
                    adj[u, v] = 1
                    adj[v, u] = 1
                    if old_state == 0:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        comp_masks[u] &= ~(1 << v)
                        comp_masks[v] &= ~(1 << u)
                    else:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                        comp_masks[u] |= (1 << v)
                        comp_masks[v] |= (1 << u)
                    current_cost += delta
                    no_improve += 1
                else:
                    no_improve += 1
        
        if is_valid(n, masks, comp_masks):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges
        if best_cost < current_cost:
            best_masks = build_masks(n, best_adj)
            best_comp = build_comp_masks(n, best_masks)
            if is_valid(n, best_masks, best_comp):
                edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
                return edges
        return None
    
    # Strategy: try to build up from 17 to 35
    current_n = 17
    current_adj = adj17.copy()
    
    while current_n < 35 and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        if remaining < 1.0:
            break
        
        target_n = current_n + 1
        budget = min(remaining * 0.25, 8.0)
        
        # Try multiple times to extend
        found = None
        num_attempts = 3
        per_attempt = budget / num_attempts
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Create seed with new vertex
            seed_adj = np.zeros((target_n, target_n), dtype=np.int8)
            seed_adj[:current_n, :current_n] = current_adj
            
            # Connect new vertex with some probability
            p_connect = 0.4 + 0.1 * random.random()
            for j in range(current_n):
                if random.random() < p_connect:
                    seed_adj[current_n, j] = 1
                    seed_adj[j, current_n] = 1
            
            found = try_repair(target_n, per_attempt, seed_adj=seed_adj)
            if found is not None:
                break
        
        if found is not None:
            current_n = target_n
            # Reconstruct adj from edges
            current_adj = np.zeros((current_n, current_n), dtype=np.int8)
            for u, v in found:
                current_adj[u, v] = 1
                current_adj[v, u] = 1
            best_n = current_n
            best_edges = found
        else:
            # Try random restart with larger n
            if current_n >= 25:
                break
    
    # Final attempt: try to reach 35 directly if we got close
    if best_n < 35 and time.time() < deadline - 3.0:
        remaining = deadline - time.time() - 3.0
        for target in range(max(best_n + 1, 30), 36):
            if time.time() > deadline - 2.0:
                break
            budget = min(remaining * 0.15, 6.0)
            found = try_repair(target, budget)
            if found is not None:
                best_n = target
                best_edges = found
                break
    
    return {'n': best_n, 'edges': best_edges}
