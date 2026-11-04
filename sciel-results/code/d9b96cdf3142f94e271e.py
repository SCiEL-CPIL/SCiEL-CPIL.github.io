<<<<<<< SEARCH
    # Try ranks from low to high
    # Start from 40, which is a reasonable lower bound for 4x5x5
    # Known bounds: rank(4x5x5) <= 4*5*5 = 100.
    # Strassen-like algorithms might get close to 2^(log2(4)+log2(5)+log2(5)) ~ 100? No.
    # Let's try a range.
    for R in range(40, 101, 5):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result is not None:
            return result
    
    return schoolbook()
=======
    # Try ranks from low to high
    # Start from 40, which is a reasonable lower bound for 4x5x5
    # Known bounds: rank(4x5x5) <= 4*5*5 = 100.
    # Let's try a finer range.
    for R in range(40, 101, 2):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result is not None:
            return result
    
    return schoolbook()
>>>>>>> REPLACE
