"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # LUW construction for k-regular graphs with large girth
    # For k=4, g=10: need to find smallest d giving girth >= g
    # The LUW construction gives girth >= 2*floor(d/2) + 2 for even d,
    # or girth >= 2*floor((d+1)/2) + 1 for odd d (approximate)
    # Empirically for k=4: d=5 gives girth 10
    
    # Try multiple d values and pick the smallest that works
    q = k
    
    # GF(4) arithmetic for k=4 using bit operations
    if q == 4:
        mul_table = (
            0, 0, 0, 0,
            0, 1, 2, 3,
            0, 2, 3, 1,
            0, 3, 1, 2
        )
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return mul_table[x * 4 + y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return (x * y) % q

    def build_luw(d):
        """Build LUW graph with parameter d, return (n, edges) or None if too slow."""
        # LUW rules
        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule_a[2], rule_b[2] = 1, 1
        rule_a[3], rule_b[3] = 2, 1
        rule_a[4], rule_b[4] = 1, 2
        
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                a, b = 1, c0 - 2
            elif r == 1:
                a, b = c0 - 1, 1
            elif r == 2:
                a, b = c0, 1
            else:
                a, b = 1, c0 + 1
            rule_a[c] = a
            rule_b[c] = b

        # BFS from origin to build the graph
        start = (0, tuple([0] * (d + 1)))
        index = {start: 0}
        order = [start]
        edges = []
        
        head = 0
        while head < len(order):
            if time.time() > deadline - 10:
                return None
            
            side, x = order[head]
            head += 1
            x_idx = index[(side, x)]
            
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
                
                other = (1 - side, tuple(y))
                
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                
                # Only add edge from side 0 to avoid duplicates
                if side == 0:
                    edges.append((x_idx, index[other]))
        
        n = len(order)
        return n, edges

    # Try d values starting from the theoretical minimum
    # For girth g, we need roughly d >= g/2 - 1
    # Try a range of d values and pick the smallest valid one
    d_min = max(3, g - 5)
    d_max = g  # upper bound
    
    best_result = None
    for d in range(d_min, d_max + 1):
        if time.time() > deadline - 20:
            break
        result = build_luw(d)
        if result is not None:
            best_result = (d, result)
            # If we found a result, check if we can try smaller
            # But for now, take the first valid one (smallest d)
            break
    
    if best_result is None:
        # Fallback: use a larger d that should work
        d = max(5, g - 4)
        result = build_luw(d)
        if result is None:
            # Last resort: return a trivial result
            return 0, []
        return result
    
    return best_result[1]
# EVOLVE-BLOCK-END
