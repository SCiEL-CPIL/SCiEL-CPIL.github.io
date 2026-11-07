import numpy as np

def construct():
    N = 10000000
    # Sieve for sigma(n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    best_err = float('inf')
    best_a, best_b = 1, 1
    
    # Tier 1: Look for |2a - 3b| = 1
    # This gives error 1/(2b). Maximizing b minimizes error.
    # Iterate b from N down to 1.
    # If found, it is the global best.
    for b in range(N, 0, -1):
        # Check a = (3b + 1) / 2
        # 3b + 1 must be even -> b must be odd
        if b % 2 == 1:
            a1 = (3 * b + 1) // 2
            if a1 <= N:
                if sigma[a1] == sigma[b]:
                    return [int(a1), int(b)]
            
            a2 = (3 * b - 1) // 2
            if a2 >= 1:
                if sigma[a2] == sigma[b]:
                    return [int(a2), int(b)]
                    
    # Tier 2: If no |k|=1 found, look for |k|=3
    # Error 3/(2b). Maximize b.
    # But we must compare with potential |k|=1 matches? No, we already searched all b for |k|=1.
    # So now we just find the max b for |k|=3.
    for b in range(N, 0, -1):
        if b % 2 == 1:
            a1 = (3 * b + 3) // 2
            if a1 <= N:
                if sigma[a1] == sigma[b]:
                    return [int(a1), int(b)]
            
            a2 = (3 * b - 3) // 2
            if a2 >= 1:
                if sigma[a2] == sigma[b]:
                    return [int(a2), int(b)]
                    
    # Tier 3: |k|=5
    for b in range(N, 0, -1):
        if b % 2 == 1:
            a1 = (3 * b + 5) // 2
            if a1 <= N:
                if sigma[a1] == sigma[b]:
                    return [int(a1), int(b)]
            
            a2 = (3 * b - 5) // 2
            if a2 >= 1:
                if sigma[a2] == sigma[b]:
                    return [int(a2), int(b)]

    # Fallback: Just find any pair with minimal error in a small window
    # This should not be reached if sigma collisions are frequent enough.
    # But to be safe, we do a general search.
    best_err = float('inf')
    best_a, best_b = 1, 1
    for b in range(1, N + 1):
        target = 1.5 * b
        a_center = int(target)
        for a in [a_center - 2, a_center - 1, a_center, a_center + 1, a_center + 2]:
            if 1 <= a <= N and a != b:
                if sigma[a] == sigma[b]:
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
                        if best_err < 1e-7: # Good enough
                            break
        if best_err < 1e-7:
            break
            
    return [int(best_a), int(best_b)]
