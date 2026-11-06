import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin for 120s limit

    # Set L
    # L = 10^9 seems reasonable for memory and time
    L = 10**9

    # 1. Generate powerful numbers <= L
    powerful = set()
    # y is square-free
    # x^2 y^3 <= L
    # y <= L^(1/3)
    max_y = int(L ** (1/3)) + 1
    
    # Precompute square-free check for y
    # y is small (<= 1000), so simple check is fine
    for y in range(1, max_y + 1):
        # Check if y is square-free
        is_sq_free = True
        temp_y = y
        d = 2
        while d * d <= temp_y:
            if temp_y % (d * d) == 0:
                is_sq_free = False
                break
            d += 1
        if not is_sq_free:
            continue
            
        y_cubed = y ** 3
        if y_cubed > L:
            break
        max_x = int(math.isqrt(L // y_cubed))
        for x in range(1, max_x + 1):
            val = x * x * y_cubed
            powerful.add(val)
            
    P = sorted(powerful)
    K = len(P)
    
    # Time check
    if time.time() > deadline:
        # Fallback to smaller L if generation was slow (unlikely)
        L = 10**8
        powerful = set()
        max_y = int(L ** (1/3)) + 1
        for y in range(1, max_y + 1):
            is_sq_free = True
            temp_y = y
            d = 2
            while d * d <= temp_y:
                if temp_y % (d * d) == 0:
                    is_sq_free = False
                    break
                d += 1
            if not is_sq_free:
                continue
            y_cubed = y ** 3
            if y_cubed > L:
                break
            max_x = int(math.isqrt(L // y_cubed))
            for x in range(1, max_x + 1):
                val = x * x * y_cubed
                powerful.add(val)
        P = sorted(powerful)
        K = len(P)

    # 2. Count sums using dictionary
    counts = {}
    P_list = P
    K_val = K
    
    # Optimization: local variables
    get = counts.get
    
    for i in range(K_val):
        pi = P_list[i]
        for j in range(i, K_val):
            s = pi + P_list[j]
            counts[s] = get(s, 0) + 1
            get = counts.get # Re-bind? No, get is a method, it doesn't change. 
            # Actually, counts.get is looked up each time. 
            # We can just use counts[s] = counts.get(s, 0) + 1
            
    # 3. Find max
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    max_count = 0
    best_N = 0
    for s, c in counts.items():
        if c > max_count:
            max_count = c
            best_N = s
            
    # 4. Retrieve pairs
    pairs = []
    powerful_set = powerful
    for a in P_list:
        b = best_N - a
        if b < a:
            continue
        if b > L: # b must be <= L since P contains numbers <= L
            break
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
