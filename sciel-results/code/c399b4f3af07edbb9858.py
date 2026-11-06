def construct():
    MOD = 1009
    MAX_VAL = 1008
    
    # 1. Identify forbidden elements (self-inverses)
    # x^2 == 1 mod 1009 => x == 1 or x == 1008
    forbidden = {1, 1008}
    
    # 2. Build constraints
    # For each x, find y = inv(x).
    # If x < y, then if we include y, we must exclude x.
    # This imposes a constraint on the start index when the end index is y.
    # constraint_at[y] = list of x's that must be excluded if y is included.
    # Actually, we just need the maximum x to exclude?
    # No, if we have multiple x's associated with y, we must exclude ALL of them.
    # So start must be > max(x1, x2, ...).
    # Wait, is it possible for multiple x's to have the same inverse y?
    # No, inverse is a function. Each y has exactly one inverse x.
    # So for each y, there is at most one x such that x*y == 1.
    # If x < y, we have a constraint (x, y).
    # If x > y, the constraint is recorded at x (since we sort by the larger endpoint).
    # So for each y, there is at most one constraint coming from the "smaller side".
    
    # Let's refine:
    # We iterate x from 1 to 1008.
    # y = inv(x).
    # If x < y:
    #    Constraint: if end >= y, start > x.
    #    We can store this as: at position y, we must push start to x+1.
    # If x > y:
    #    Constraint: if end >= x, start > y.
    #    We can store this as: at position x, we must push start to y+1.
    
    # So, we can create an array `min_start_bound` of size 1009, initialized to 1.
    # For each x:
    #    y = inv(x)
    #    if x < y:
    #        min_start_bound[y] = max(min_start_bound[y], x + 1)
    #    elif x > y:
    #        min_start_bound[x] = max(min_start_bound[x], y + 1)
    #    else: # x == y (self-inverse)
    #        # handled by forbidden set
    
    # Then, we do a prefix max on min_start_bound?
    # No, the constraint is: for end e, start must be > all x_i where (x_i, e_i) is an edge and e_i <= e.
    # So we need `prefix_max_bound[e] = max(min_start_bound[1], ..., min_start_bound[e])`.
    
    # Let's trace this.
    # Edge (a, b) with a < b.
    # Constraint applies for all e >= b.
    # So for e = b, start > a.
    # For e = b+1, start > a.
    # ...
    # So `min_start_bound[b]` should be at least `a+1`.
    # And `prefix_max_bound[e]` will carry this forward.
    
    # This is O(N). Much faster and cleaner.

    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    # min_start_bound[i] represents the minimum start index required
    # due to conflict edges ending exactly at i.
    # We initialize it to 1.
    min_start_bound = [1] * (MOD + 1)
    
    for x in range(1, MOD):
        y = inv[x]
        if x < y:
            # Edge (x, y). If interval ends at y or later, it cannot contain x.
            # So start must be > x.
            if x + 1 > min_start_bound[y]:
                min_start_bound[y] = x + 1
        elif x > y:
            # Edge (y, x). If interval ends at x or later, it cannot contain y.
            # So start must be > y.
            if y + 1 > min_start_bound[x]:
                min_start_bound[x] = y + 1
        # else x == y, ignore (handled by forbidden)
        
    # Now compute prefix max of min_start_bound
    # prefix_bound[e] = max(min_start_bound[1], ..., min_start_bound[e])
    # This represents the minimum start index required to avoid all conflict edges
    # with larger endpoint <= e.
    
    best_start = 1
    best_length = 0
    current_min_start = 1
    
    # We also need to handle forbidden elements.
    # If e is forbidden, no interval ending at e is valid.
    # If f < e is forbidden, start must be > f.
    # We can incorporate this into the sweep.
    
    for e in range(1, MOD):
        # Update current_min_start with conflict constraints up to e
        if min_start_bound[e] > current_min_start:
            current_min_start = min_start_bound[e]
            
        # Update current_min_start with forbidden constraints
        if e in forbidden:
            # If e is forbidden, we cannot end at e?
            # Wait. If e is forbidden, the interval [s, e] contains e.
            # So [s, e] is invalid.
            # So we should not update best_length for this e.
            # But we still need to update current_min_start?
            # If e is forbidden, any interval ending at e is invalid.
            # However, for future ends e' > e, the constraint "start > e" applies
            # because e is in [s, e'] if s <= e.
            # So if we include e in the interval, it's invalid.
            # So for any valid interval ending at e' > e, we must have s > e.
            # So current_min_start must be at least e + 1.
            current_min_start = max(current_min_start, e + 1)
            # And we skip checking length for this e because it's invalid.
            continue
            
        # Check length for this e
        if current_min_start <= e:
            length = e - current_min_start + 1
            if length > best_length:
                best_length = length
                best_start = current_min_start
                
    return [best_start, best_length]
