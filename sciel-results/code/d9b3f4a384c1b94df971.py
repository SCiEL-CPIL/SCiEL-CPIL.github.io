    # ... after LUW ...
    
    # If k=5 and g=9, try to find the known cage size 1116
    if k == 5 and g == 9:
        target_n = 1116
        # Only try if current best is worse (larger) or None
        if best is None or best[0] > target_n:
            result = _try_random_regular(k, g, target_n, trials=500)
            if result is not None:
                best = result
