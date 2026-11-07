def max_size_consecutive():
    max_m = 0
    best_set = []
    for m in range(1, 61):
        # Calculate max D(k)
        max_d = -1
        for k in range(1, m):
            # Sum of top k y's: y_{m-k+1} to y_m are m-k to m-1
            sum_top = sum(range(m-k, m))
            # Sum of bottom k+1 y's: y_1 to y_{k+1} are 0 to k
            sum_bot = sum(range(0, k+1))
            d = sum_top - sum_bot
            if d > max_d:
                max_d = d
        
        a_min = max_d + 1
        # a must be at least 1
        if a_min < 1:
            a_min = 1
            
        x_max = a_min + m - 1
        if x_max <= 256:
            max_m = m
            best_set = list(range(a_min, a_min + m))
        else:
            # Since a_min grows with m, if it fails here, it might fail for larger m?
            # a_min ~ m^2/4, x_max ~ m^2/4 + m. Monotonic increasing.
            # So we can break.
            break
    return best_set
