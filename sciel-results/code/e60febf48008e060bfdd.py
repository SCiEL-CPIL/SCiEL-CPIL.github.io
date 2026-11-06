import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # Limit for powerful numbers
    # 10^9 gives ~38k numbers, ~700M pairs. Might be too slow in pure Python.
    # 5*10^8 gives ~26k numbers, ~300M pairs. Safer.
    # Let's try 10^9 with time checking.
    L = 10**9
    
    powerful = set()
    # y^3 <= L => y <= 1000
    for y in range(1, 1001):
        y3 = y ** 3
        if y3 > L:
            break
        # x^2 <= L / y^3 => x <= sqrt(L / y^3)
        max_x = int((L // y3) ** 0.5)
        for x in range(1, max_x + 1):
            val = x * x * y3
            if val <= L:
                powerful.add(val)
            else:
                break
    
    P = sorted(powerful)
    n = len(P)
    
    counts = {}
    best_N = 0
    best_count = 0
    
    # Optimization: If n is too large, we might TLE.
    # n ~ 38000. n^2/2 ~ 722,000,000.
    # In Python, 1 loop iteration is ~50-100ns? No, more like 100ns for simple ops, but dict update is slower.
    # ~100ns * 7e8 = 70 seconds.
    # It's risky. Let's use 5*10^8 to be safe, or add time check.
    
    # If we use 5*10^8, n ~ 26000. Pairs ~ 338M. Time ~ 30-40s.
    # Let's stick to 10^9 but check time every 10000 iterations.
    
    check_interval = 10000
    iter_count = 0
    
    for i in range(n):
        ai = P[i]
        # b >= a, so j starts at i
        for j in range(i, n):
            s = ai + P[j]
            # Update count
            c = counts.get(s, 0) + 1
            counts[s] = c
            if c > best_count:
                best_count = c
                best_N = s
            
            iter_count += 1
            if iter_count % check_interval == 0:
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break
    
    # Extract pairs for best_N
    pairs = []
    for i in range(n):
        ai = P[i]
        for j in range(i, n):
            if ai + P[j] == best_N:
                pairs.append([ai, P[j]])
    
    if not pairs:
        # Fallback if something went wrong or no pairs found (should not happen)
        # Use a simple known powerful sum
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': pairs}
