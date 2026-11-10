def _is_valid_desc(S, x):
    """
    S contains elements all > x.
    Check if any a in S is mean of subset of S U {x} \ {a}.
    """
    n = len(S)
    if n < 2:
        return True
    
    # We need to check for each a in S
    # Is a the mean of any subset of V = S \ {a} U {x} of size >= 2?
    
    for a in S:
        # V = S \ {a} U {x}
        # We need to check if sum(T) = k*a for any T subset of V, |T|=k>=2.
        
        # Build DP for V
        # V has n elements.
        dp_V = [0] * (n + 1)
        dp_V[0] = 1
        
        # Add S \ {a}
        for elem in S:
            if elem == a:
                continue
            for k in range(n, 0, -1):
                dp_V[k] |= (dp_V[k-1] << elem)
        
        # Add x
        for k in range(n, 0, -1):
            dp_V[k] |= (dp_V[k-1] << x)
            
        max_sum_V = sum(V) # Can precompute
        
        for k in range(2, n + 1):
            target = k * a
            if target <= max_sum_V:
                if (dp_V[k] >> target) & 1:
                    return False

    return True
