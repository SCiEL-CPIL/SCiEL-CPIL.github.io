import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if k != 4:
        return _construct_fallback(k, g)
    
    if g == 10:
        result = _try_cayley_z2z2z5z16_optimized(deadline)
        if result:
            return result
            
    # Fallback
    return _construct_fallback(k, g)

def _try_cayley_z2z2z5z16_optimized(deadline):
    # Group G = Z_2 x Z_2 x Z_5 x Z_16
    # Map element (a,b,c,d) to index: a + 2b + 4c + 20d
    # Total 320 elements
    
    # Pre-compute group operations
    # indices 0..319
    add_table = np.zeros((320, 320), dtype=np.int32)
    neg_table = np.zeros(320, dtype=np.int32)
    
    def idx_to_elem(i):
        a = i % 2
        b = (i // 2) % 2
        c = (i // 4) % 5
        d = (i // 20) % 16
        return (a, b, c, d)
        
    def elem_to_idx(e):
        return e[0] + 2*e[1] + 4*e[2] + 20*e[3]
        
    # Build tables
    for i in range(320):
        ei = idx_to_elem(i)
        neg_table[i] = elem_to_idx(((-ei[0])%2, (-ei[1])%2, (-ei[2])%5, (-ei[3])%16))
        for j in range(320):
            ej = idx_to_elem(j)
            s = ((ei[0]+ej[0])%2, (ei[1]+ej[1])%2, (ei[2]+ej[2])%5, (ei[3]+ej[3])%16)
            add_table[i, j] = elem_to_idx(s)
            
    # Pre-compute forbidden exponent pairs (k1, k2) for non-backtracking words of length 2 to 9
    # Symbols: 1, -1, 2, -2
    # Non-backtracking: next != -prev
    
    forbidden_pairs = set()
    
    def gen_words(prefix, k1, k2, length, max_len):
        if length == 0:
            if len(prefix) >= 2:
                forbidden_pairs.add((k1, k2))
            return
        
        last = prefix[-1] if prefix else None
        for sym in [1, -1, 2, -2]:
            if last is not None and sym == -last:
                continue
            nk1 = k1 + (1 if sym == 1 else -1 if sym == -1 else 0)
            nk2 = k2 + (1 if sym == 2 else -1 if sym == -2 else 0)
            gen_words(prefix + [sym], nk1, nk2, length - 1, max_len)
            
    for L in range(2, 10):
        for sym in [1, -1, 2, -2]:
            k1 = 1 if sym == 1 else -1 if sym == -1 else 0
            k2 = 1 if sym == 2 else -1 if sym == -2 else 0
            gen_words([sym], k1, k2, L - 1, 9)
            
    # List of unique k1 and k2 values
    k1_vals = list(set(k1 for k1, k2 in forbidden_pairs))
    k2_vals = list(set(k2 for k1, k2 in forbidden_pairs))
    
    # Map k1 to index in k1_vals for fast lookup? No, just iterate.
    # For each s1, compute k1*s1 for all k1 in k1_vals
    # For each s2, compute k2*s2 for all k2 in k2_vals
    # Check intersection of { -k1*s1 } and { k2*s2 }
    
    # To speed up, for a fixed s1, compute set1 = { add_table[ k1_s1 ] [ 0 ] ... }? No.
    # val1[k1] = k1 * s1
    # val2[k2] = k2 * s2
    # Check if any val1[k1] + val2[k2] == 0 => val2[k2] == -val1[k1]
    
    # Pre-compute powers of s1 and s2?
    # Multiplication by integer k in abelian group is repeated addition.
    # We can pre-compute k*element for all k in k1_vals and k2_vals for all elements?
    # 320 elements * ~10 k values = 3200 ops. Fast.
    
    # Let's pre-compute mult_table[k_val][element_idx] = k_val * element_idx
    # But k_vals are different for k1 and k2.
    # Let's just compute on the fly for each s1, s2 pair, but use the add_table.
    
    # Actually, we can pre-compute for each element e, the value k*e for all k in k1_vals.
    # Let's create a dictionary or list.
    
    # Optimization:
    # For a fixed s1, compute V1 = { (k1, k1*s1) for k1 in k1_vals }
    # For a fixed s2, compute V2 = { (k2, k2*s2) for k2 in k2_vals }
    # We need to check if there exists (k1, v1) in V1 and (k2, v2) in V2 such that
    # (k1, k2) is in forbidden_pairs AND v1 + v2 == 0.
    
    # Since forbidden_pairs is small (~20k), and V1, V2 are small (~10 each),
    # we can iterate over forbidden_pairs.
    
    # Better:
    # For fixed s1, build a set BadVals1 = { -k1*s1 for (k1, k2) in forbidden_pairs }
    # For fixed s2, build a set GoodVals2 = { k2*s2 for (k1, k2) in forbidden_pairs }
    # If BadVals1 & GoodVals2 is not empty, invalid.
    
    # Wait, the condition is: for ALL (k1, k2) in forbidden_pairs, k1*s1 + k2*s2 != 0.
    # This is equivalent to: The set { -k1*s1 | (k1, k2) in FP } is disjoint from { k2*s2 | (k1, k2) in FP }?
    # No. It's a pairwise condition.
    # It is NOT enough that the sets are disjoint.
    # Example: k1=1, k2=1. k1'=1, k2'=2.
    # We need 1*s1 + 1*s2 != 0 AND 1*s1 + 2*s2 != 0.
    # If we just check sets, we might miss the correlation.
    
    # So we must iterate over the pairs in forbidden_pairs.
    
    # To make this fast:
    # For fixed s1, pre-compute k1*s1 for all k1 in k1_vals. Store in dict or list.
    # For fixed s2, pre-compute k2*s2 for all k2 in k2_vals.
    # Then iterate over forbidden_pairs.
    
    # Number of pairs (s1, s2) is ~100k.
    # Iterating 20k pairs for each is 2e9. Too slow.
    
    # We need to prune s1 or s2.
    # Note: s1 and s2 are symmetric.
    # We can fix s1 to be the smallest index in its "orbit" under some symmetry?
    # Or just try s1=1, 2, ... and for each s1, try s2=1, 2...
    # If we find a valid pair, we stop.
    # The solution exists. It might be found early.
    
    # Let's try to optimize the inner loop with NumPy?
    # No, the logic is discrete.
    
    # Let's try a different approach:
    # Random sampling? No, must be deterministic for reproducibility?
    # The evaluator doesn't require determinism, but it's good practice.
    
    # Let's stick to the deterministic search but add a timeout.
    # If we don't find it in 110s, return fallback.
    
    # To speed up the inner check:
    # Pre-compute for each s1:
    #   val1_dict: k1 -> k1*s1
    # For each s2:
    #   val2_dict: k2 -> k2*s2
    # Check:
    #   for (k1, k2) in forbidden_pairs:
    #      if add_table[ val1_dict[k1] ][ val2_dict[k2] ] == 0: return False
    
    # This is still 20k ops per pair.
    
    # Can we reduce forbidden_pairs?
    # Many pairs (k1, k2) are redundant?
    # No, each word is distinct.
    
    # Let's try to limit the search.
    # Maybe the solution uses small elements?
    
    # We will iterate s1 from 1 to 319.
    # For each s1, iterate s2 from 1 to 319.
    # Check time every 1000 pairs.
    
    # To make it faster, we can pre-compute the "bad" s2 values for a given s1?
    # No, s2 is a variable.
    
    # Let's just run it and hope it finds a solution quickly.
    # The known solution might be with small indices.
    
    current_time = time.time()
    
    for s1 in range(1, 320):
        if time.time() > deadline:
            return None
            
        # Pre-compute k1*s1 for all k1 in k1_vals
        val1 = {}
        for k1 in k1_vals:
            # Compute k1 * s1
            res = 0
            if k1 > 0:
                for _ in range(k1):
                    res = add_table[res, s1]
            elif k1 < 0:
                neg_s1 = neg_table[s1]
                for _ in range(-k1):
                    res = add_table[res, neg_s1]
            val1[k1] = res
            
        for s2 in range(1, 320):
            if s2 == s1 or s2 == neg_table[s1] or s1 == 0 or s2 == 0:
                continue
                
            # Pre-compute k2*s2 for all k2 in k2_vals
            val2 = {}
            for k2 in k2_vals:
                res = 0
                if k2 > 0:
                    for _ in range(k2):
                        res = add_table[res, s2]
                elif k2 < 0:
                    neg_s2 = neg_table[s2]
                    for _ in range(-k2):
                        res = add_table[res, neg_s2]
                val2[k2] = res
                
            # Check constraints
            valid = True
            for k1, k2 in forbidden_pairs:
                v1 = val1[k1]
                v2 = val2[k2]
                if add_table[v1, v2] == 0:
                    valid = False
                    break
                    
            if valid:
                # Found valid s1, s2
                # Construct edges
                edges = []
                for v in range(320):
                    n1 = add_table[v, s1]
                    n2 = add_table[v, neg_table[s1]]
                    n3 = add_table[v, s2]
                    n4 = add_table[v, neg_table[s2]]
                    
                    for n in [n1, n2, n3, n4]:
                        if v < n:
                            edges.append((v, n))
                            
                return 320, sorted(edges)
                
    return None

def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed."""
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
