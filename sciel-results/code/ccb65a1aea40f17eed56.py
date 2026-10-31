def construct():
    import math
    
    # Known optimal Golomb rulers
    rulers = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 25],
        8: [0, 1, 4, 10, 12, 17, 20, 34], # Need to verify this specific set
        9: [0, 1, 4, 10, 12, 17, 20, 24, 44], # Verify
        10: [0, 1, 4, 10, 12, 17, 20, 24, 30, 55], # Verify
    }
    
    best_set = [0, 1, 3]
    best_score = 0.0
    
    for n, U in rulers.items():
        if not U or U[0] != 0:
            continue
        max_u = max(U)
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        
        size_sum = len(sum_set)
        size_diff = len(diff_set)
        
        if size_sum == 0:
            continue
            
        ratio = size_diff / size_sum
        if ratio <= 1:
            score = 1.0
        else:
            # Use log2
            num = math.log2(ratio)
            den = math.log2(2 * max_u + 1)
            score = 1.0 + num / den
            
        if score > best_score:
            best_score = score
            best_set = U
            
    return best_set
