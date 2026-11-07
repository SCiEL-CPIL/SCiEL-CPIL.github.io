def construct():
    L = 10 * 10**6
    
    sigma = [1] * (L + 1)
    for i in range(1, L + 1):
        for j in range(i, L + 1, i):
            sigma[j] += i
            
    MAX_M = 10**18
    counts = {}
    
    # First pass: count
    for k in range(1, L + 1):
        m = k * sigma[k]
        if m <= MAX_M:
            counts[m] = counts.get(m, 0) + 1
            
    # Find best M
    best_M = 1
    best_count = 0
    for m, c in counts.items():
        if c > best_count:
            best_count = c
            best_M = m
            
    # Second pass: collect k
    best_k = []
    for k in range(1, L + 1):
        m = k * sigma[k]
        if m == best_M:
            best_k.append(k)
            
    return {'M': best_M, 'k': best_k}
