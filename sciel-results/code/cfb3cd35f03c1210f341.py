def construct():
    N = 256
    S = set(range(1, N + 1))
    
    # Iterate from largest to smallest
    # If an element is in a 3-AP, remove it.
    # This prioritizes keeping small numbers (high reciprocal weight).
    for x in range(N, 0, -1):
        if x not in S:
            continue
            
        is_in_ap = False
        
        # Check if x is the largest term: a, b, x
        # b = x - d, a = x - 2d
        # d >= 1, a >= 1 => x - 2d >= 1 => d <= (x-1)//2
        max_d_end = (x - 1) // 2
        for d in range(1, max_d_end + 1):
            if (x - d) in S and (x - 2 * d) in S:
                is_in_ap = True
                break
        
        # Check if x is the middle term: a, x, c
        # a = x - d, c = x + d
        # c <= N => x + d <= N => d <= N - x
        max_d_mid = N - x
        if not is_in_ap:
            for d in range(1, max_d_mid + 1):
                if (x - d) in S and (x + d) in S:
                    is_in_ap = True
                    break
                    
        # Check if x is the smallest term: x, b, c
        # b = x + d, c = x + 2d
        # c <= N => x + 2d <= N => d <= (N - x) // 2
        max_d_start = (N - x) // 2
        if not is_in_ap:
            for d in range(1, max_d_start + 1):
                if (x + d) in S and (x + 2 * d) in S:
                    is_in_ap = True
                    break
                    
        if is_in_ap:
            S.remove(x)
            
    return list(S)
