import math
import itertools
import time
import bisect

def construct():
    start_time = time.time()
    timeout = 85.0

    # 1. Calculate LCM of 1 to 30
    L = 1
    for i in range(1, 31):
        L = math.lcm(L, i)
        
    # Weights w_i = L / i
    weights = [L // i for i in range(1, 31)]
    
    w1 = weights[:15]
    w2 = weights[15:]
    
    # 2. Generate all sums for first half
    data_A = []
    for combo in itertools.product([-1, 0, 1], repeat=15):
        s = 0
        code = 0
        for i in range(15):
            c = combo[i]
            s += c * w1[i]
            code = code * 3 + (c + 1)
        data_A.append((s, code))
        
    data_A.sort(key=lambda x: x[0])
    
    sums_A = [x[0] for x in data_A]
    
    best_abs_sum = float('inf')
    best_code_A = 0
    best_code_B = 0
    
    for combo_b in itertools.product([-1, 0, 1], repeat=15):
        s_b = 0
        code_b = 0
        for j in range(15):
            c = combo_b[j]
            s_b += c * w2[j]
            code_b = code_b * 3 + (c + 1)
            
        target = -s_b
        idx = bisect.bisect_left(sums_A, target)
        
        # Check neighbors to handle zero sum exclusion and closest match
        for k in [idx - 1, idx, idx + 1]:
            if 0 <= k < len(data_A):
                s_a, code_a = data_A[k]
                total_sum = s_a + s_b
                if total_sum != 0:
                    abs_total = abs(total_sum)
                    if abs_total < best_abs_sum:
                        best_abs_sum = abs_total)
                        best_code_A = code_a
                        best_code_B = code_b
                        
        if time.time() - start_time > timeout:
            break

    # Reconstruct
    coeffs = [0] * 30
    
    code_a = best_code_A
    for i in range(14, -1, -1):
        coeffs[i] = (code_a % 3) - 1
        code_a //= 3
        
    code_b = best_code_B
    for i in range(14, -1, -1):
        coeffs[15 + i] = (code_b % 3) - 1
        code_b //= 3

    return coeffs
