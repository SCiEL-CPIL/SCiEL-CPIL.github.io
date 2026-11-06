def construct():
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
        
    A = list(range(1, 6))
    B = list(range(6, 11))
    
    # Build dict for A: sum -> (min_size, mask)
    dict_A = {}
    for mask in range(1 << len(A)):
        s = 0
        for i, x in enumerate(A):
            if mask & (1 << i):
                s = (s + inv[x]) % 101
        sz = bin(mask).count("1")
        if s not in dict_A or sz < dict_A[s][0]:
            dict_A[s] = (sz, mask)
            
    # Build dict for B: sum -> (min_size, mask)
    dict_B = {}
    for mask in range(1 << len(B)):
        s = 0
        for i, x in enumerate(B):
            if mask & (1 << i):
                s = (s + inv[x]) % 101
        sz = bin(mask).count("1")
        if s not in dict_B or sz < dict_B[s][0]:
            dict_B[s] = (sz, mask)
            
    best = {}
    for r in range(101):
        min_total_size = 11
        best_mask = 0
        
        for sum_a, (size_a, mask_a) in dict_A.items():
            target_b = (r - sum_a) % 101
            if target_b in dict_B:
                size_b, mask_b = dict_B[target_b]
                total_size = size_a + size_b
                if total_size > 0 and total_size < min_total_size:
                    min_total_size = total_size
                    best_mask = mask_a | (mask_b << len(A))
                    
        # Reconstruct
        terms = []
        for i, x in enumerate(A):
            if best_mask & (1 << i):
                terms.append(x)
        for i, x in enumerate(B):
            if best_mask & (1 << (i + len(A))):
                terms.append(x)
        best[r] = terms
        
    return [best[r] for r in range(101)]
