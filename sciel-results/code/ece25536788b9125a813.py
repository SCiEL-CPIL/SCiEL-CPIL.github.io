def construct():
    import time
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin
    
    base = set(range(91, 181))
    result = set(base)
    
    # Precompute: for each x in [1,90], check if it can be added
    # We'll greedily add from largest to smallest (larger x are "safer" as 1/x is smaller)
    
    def is_valid_add(x, current_set):
        """Check if adding x to current_set creates any forbidden triple."""
        # Check all pairs (u, v) in current_set, u != v, u != x, v != x
        # Such that {x, u, v} is a forbidden triple
        for u in current_set:
            if u == x:
                continue
            for v in current_set:
                if v == x or v == u:
                    continue
                # Check if 1/a = 1/b + 1/c for some permutation of (x, u, v)
                # a is the smallest
                a, b, c = sorted([x, u, v])
                # Check 1/a = 1/b + 1/c
                # Equivalent: a = bc/(b+c), i.e., a*(b+c) == b*c
                if a * (b + c) == b * c:
                    return False
        return True
    
    # This is O(n^2) per addition, which is too slow for 90 additions with ~180 elements
    # Let's be smarter
    
    # For adding x, forbidden triples involving x:
    # 1/x = 1/u + 1/v: x = uv/(u+v), so u,v must satisfy uv = x(u+v) => uv - xu - xv = 0 => (u-x)(v-x) = x^2
    # 1/u = 1/x + 1/v: u = xv/(x+v), so v = ux/(u-x) for u > x
    # 1/v = 1/x + 1/u: v = xu/(x+u) for u > x
    
    # Better: precompute conflicts
    # For each x in [1,90], find all pairs (u,v) in [1,180] that form a forbidden triple with x
    
    # Let's just do it efficiently with the structure
    # Start with base, try to add each x in [1,90] in some order
    
    # Greedy: try adding from 90 down to 1
    for x in range(90, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        if is_valid_add(x, result):
            result.add(x)
    
    return list(result)
