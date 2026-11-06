def construct():
    MOD = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 1
    best_length = 0
    
    for s in range(1, n + 1):
        # Find maximum e such that [s, e] is valid
        # [s, e] valid iff for all x in [s, e], inv[x] < s or inv[x] > e
        # For fixed s, as e increases, the constraint tightens.
        # Let's find the max e by checking incrementally or by computing the bound.
        
        # For fixed s, the condition is: for all x in [s, e], if inv[x] >= s then inv[x] > e.
        # So e must be less than min{ inv[x] : x in [s, e], inv[x] >= s }
        # But this depends on e.
        
        # Alternative: for fixed s, define g(e) = min{ inv[x] : x in [s, e], inv[x] >= s }
        # Then [s, e] is valid iff e < g(e), i.e., e < min{ inv[x] : x in [s, e], inv[x] >= s }
        
        # Let's just scan e from s to n and check.
        max_e = s - 1
        for e in range(s, n + 1):
            # Check if adding e keeps validity
            # [s, e] valid iff for all x in [s, e], inv[x] < s or inv[x] > e
            # We can maintain a running check.
            # But O(n²) total is fine.
            valid = True
            for x in range(s, e + 1):
                if s <= inv[x] <= e:
                    valid = False
                    break
            if valid:
                max_e = e
            else:
                # Once invalid, larger e will also be invalid? Not necessarily,
                # because the condition is "for all x in [s,e]". If [s,e] is invalid,
                # [s, e+1] is also invalid because it contains [s,e].
                break
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
    
    return [best_start, best_length]
