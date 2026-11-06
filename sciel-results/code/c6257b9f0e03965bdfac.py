def construct():
    n = 192
    max_size = 48
    
    def is_safe(a, S_list):
        others = [x for x in S_list if x != a]
        if len(others) < 3:
            return True
        
        # Sort to easily get min/max sums
        others_sorted = sorted(others)
        min_sum = others_sorted[0] + others_sorted[1] + others_sorted[2]
        max_sum = others_sorted[-1] + others_sorted[-2] + others_sorted[-3]
        
        k_min = (min_sum + a - 1) // a
        k_max = max_sum // a
        
        if k_min > k_max:
            return True
            
        O_set = set(others)
        
        for k in range(k_min, k_max + 1):
            target = k * a
            # Check if target can be formed by 3 distinct elements in others
            # Iterate pairs (c, d)
            for i in range(len(others)):
                c = others[i]
                # Optimization: if c + others_sorted[0] + others_sorted[1] > target, break?
                # Not strictly necessary for N=47.
                for j in range(i + 1, len(others)):
                    d = others[j]
                    b = target - c - d
                    if b in O_set and b != c and b != d:
                        return False
        return True

    def is_valid(S_list):
        for a in S_list:
            if not is_safe(a, S_list):
                return False
        return True

    # Initial set: largest 48 numbers
    S = list(range(n - max_size + 1, n + 1))
    
    # Repair loop
    for _ in range(200):
        violator = None
        for a in S:
            if not is_safe(a, S):
                violator = a
                break
        
        if violator is None:
            break
        
        # Try to replace violator with a number outside S
        candidates = [x for x in range(n, 0, -1) if x not in S]
        swapped = False
        for y in candidates:
            S_new = [y if x == violator else x for x in S]
            if is_valid(S_new):
                S = S_new
                swapped = True
                break
        
        if not swapped:
            break
            
    return S
