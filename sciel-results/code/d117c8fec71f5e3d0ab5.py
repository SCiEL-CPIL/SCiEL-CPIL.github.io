def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute gcd table
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Precompute bad_with[u][v]: list of x such that {u,v,x} is a bad triple.
    # Note: x must be distinct from u and v.
    bad_with = [[[] for _ in range(n+1)] for __ in range(n+1)]
    for u in range(1, n+1):
        for v in range(u+1, n+1):
            d = gcd_table[u][v]
            # x must satisfy gcd(u,x) == d and gcd(v,x) == d
            for x in range(1, n+1):
                if x == u or x == v:
                    continue
                if gcd_table[u][x] == d and gcd_table[v][x] == d:
                    bad_with[u][v].append(x)
                    bad_with[v][u].append(x)
                    
    # Initialize state
    # S is a bitmask
    S = 0
    # conflict_count[x] = number of pairs (u,v) in S such that {u,v,x} is bad
    conflict_count = [0]*(n+1)
    
    def can_add(x):
        return conflict_count[x] == 0
        
    def add(x):
        nonlocal S
        S |= (1 << x)
        # Update conflict counts for all y not in S?
        # Actually, we only care about conflict_count for y NOT in S when checking addition.
        # But when we add x, it creates new pairs (x, u) for all u in S.
        # These new pairs may create conflicts for other y.
        # So we must update conflict_count[y] for all y.
        # For each u in S (excluding x, but x is new so all u in old S):
        #   The pair (x, u) is added to S.
        #   For each y in bad_with[x][u], if y is not in S, increment conflict_count[y].
        #   Also, if y is in S, we have a problem? No, S is always valid.
        #   Wait, if y is in S, then {x, u, y} is a bad triple, which would mean S was invalid before adding x?
        #   No, we check can_add(x) before adding. If can_add(x) is true, then conflict_count[x] == 0.
        #   This means there are no pairs (u,v) in S such that {u,v,x} is bad.
        #   So adding x is safe.
        #   Now, adding x creates new pairs (x, u).
        #   For any y NOT in S, if {x, u, y} is bad, conflict_count[y] increases.
        #   For any y IN S, {x, u, y} cannot be bad, because if it were, conflict_count[x] would have been > 0 (since (u,y) is a pair in S).
        
        for u_bit in range(1, n+1):
            if (S & (1 << u_bit)) and u_bit != x:
                for y in bad_with[x][u_bit]:
                    if not (S & (1 << y)):
                        conflict_count[y] += 1
                        
    def remove(x):
        nonlocal S
        # Remove x from S
        S &= ~(1 << x)
        # Update conflict counts.
        # Pairs (x, u) for all u in S (new S, so u != x) are removed.
        # For each u in S:
        #   For each y in bad_with[x][u], if y not in S, decrement conflict_count[y].
        for u_bit in range(1, n+1):
            if (S & (1 << u_bit)):
                for y in bad_with[x][u_bit]:
                    if not (S & (1 << y)):
                        conflict_count[y] -= 1

    # Greedy Initialization
    # Order: try to add numbers with low "potential" conflicts first?
    # Or just random.
    # Let's try a few random restarts.
    
    best_S = 0
    best_size = 0
    
    random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline - 10:
            break
            
        S = 0
        conflict_count = [0]*(n+1)
        
        # Shuffle numbers
        order = list(range(1, n+1))
        random.shuffle(order)
        
        for x in order:
            if can_add(x):
                add(x)
                
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_S = S
            best_conflict = conflict_count[:]
            
    # Local Search on Best
    if best_S:
        S = best_S
        conflict_count = best_conflict
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add any element
            for x in range(1, n+1):
                if not (S & (1 << x)):
                    if can_add(x):
                        add(x)
                        improved = True
                        break
                        
            if improved:
                continue
                
            # 2. Try 1-for-1 swap: remove u, add x
            # To speed up, we can check if removing u allows adding x.
            # Removing u decreases conflict_count[x] by the number of pairs (u,v) in S\{u} such that {u,v,x} is bad?
            # No, conflict_count[x] counts pairs in S.
            # If we remove u, the pairs involving u are gone.
            # Let delta = number of v in S\{u} such that {u,v,x} is bad.
            # Then new_conflict_count[x] = conflict_count[x] - delta.
            # We can precompute for each u, x, the delta?
            # delta(u, x) = count of v in S such that v != u and {u,v,x} is bad.
            # This is equivalent to: for each v in S, check if v in bad_with[u][x].
            # Since bad_with[u][x] is a list, we can just iterate and check if v is in S.
            
            found_swap = False
            # Iterate u in S
            for u in range(1, n+1):
                if not (S & (1 << u)):
                    continue
                # Calculate delta for each x?
                # Or just try to remove u, check if any x can be added, then re-add u?
                # That is expensive.
                
                # Let's compute delta for all x not in S.
                # delta[x] = 0
                # For each v in S (v != u):
                #   if x in bad_with[u][v]: delta[x] += 1
                
                # Instead, let's just simulate removal of u.
                # Save state?
                # Or just calculate if there exists x such that conflict_count[x] - delta(u,x) == 0.
                
                # Optimization:
                # Precompute for each u, a list of (v, x) pairs?
                # Let's just do the O(K^2) check for each u.
                # K ~ 100. K^2 = 10000.
                # For each u, we check all x.
                # Total O(K^3) per iteration? 100^3 = 10^6.
                # With 50 iterations, 5*10^7. Fast enough.
                
                # Let's try a specific x first?
                # Just try all x for each u.
                
                # To avoid O(K^2) for each x, we can precompute the set of x that are "blocked" by u.
                # Actually, we can just remove u, check all x, and if none work, re-add u.
                
                # Save S and conflict_count
                old_S = S
                old_cc = conflict_count[:]
                
                remove(u)
                
                added_x = -1
                for x in range(1, n+1):
                    if not (S & (1 << x)):
                        if can_add(x):
                            add(x)
                            added_x = x
                            break
                            
                if added_x != -1:
                    # Swap successful
                    improved = True
                    found_swap = True
                    break
                else:
                    # Revert
                    S = old_S
                    conflict_count = old_cc
                    # Note: remove(u) modified S and cc. We restored them.
                    # But wait, remove(u) changed S. We set S = old_S.
                    # And conflict_count = old_cc.
                    # This is correct.
            
            if found_swap:
                continue
                
            # 3. Try 1-for-2 swap: remove u, add x1, x2
            # This is more expensive.
            # Try to remove u. Then try to add x1. Then try to add x2.
            # If both added, size increases by 1.
            
            found_swap2 = False
            for u in range(1, n+1):
                if not (S & (1 << u)):
                    continue
                if time.time() > deadline - 5:
                    break
                    
                old_S = S
                old_cc = conflict_count[:]
                
                remove(u)
                
                added_count = 0
                added_elements = []
                
                # Try to add as many as possible
                for x in range(1, n+1):
                    if not (S & (1 << x)):
                        if can_add(x):
                            add(x)
                            added_count += 1
                            added_elements.append(x)
                            if added_count == 2:
                                break
                                
                if added_count >= 2:
                    improved = True
                    found_swap2 = True
                    break
                else:
                    # Revert
                    S = old_S
                    conflict_count = old_cc
            
            if found_swap2:
                continue
            
    result = [i for i in range(1, n+1) if (best_S >> i) & 1]
    if not result:
        result = [1]
        
    return result
