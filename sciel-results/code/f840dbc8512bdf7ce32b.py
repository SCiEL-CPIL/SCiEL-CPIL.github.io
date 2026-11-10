def recover_coeffs(weights, target):
    m = len(weights)
    if m == 0:
        return []
    mid = m // 2
    w_left = weights[:mid]
    w_right = weights[mid:]
    
    # Enumerate left: map sum -> coefficient tuple
    left_sums = {0: (0,)*mid}
    for i, w in enumerate(w_left):
        new_sums = {}
        for s, coeffs in left_sums.items():
            for c in [-1, 0, 1]:
                ns = s + c * w
                new_coeffs = list(coeffs)
                new_coeffs[i] = c
                new_coeffs = tuple(new_coeffs)
                if ns not in new_sums:
                    new_sums[ns] = new_coeffs
        left_sums = new_sums
    
    # Enumerate right
    right_sums = {0: (0,)*len(w_right)}
    for i, w in enumerate(w_right):
        new_sums = {}
        for s, coeffs in right_sums.items():
            for c in [-1, 0, 1]:
                ns = s + c * w
                new_coeffs = list(coeffs)
                new_coeffs[i] = c
                new_coeffs = tuple(new_coeffs)
                if ns not in new_sums:
                    new_sums[ns] = new_coeffs
        right_sums = new_sums
    
    # Find combination
    for ls, lc in left_sums.items():
        rs = target - ls
        if rs in right_sums:
            return list(lc) + list(right_sums[rs])
    return None
