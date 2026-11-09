import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    D = 1
    for i in range(1, 31):
        D = lcm(D, i)
        
    # 2. Generate sums for first 15 terms
    # Coefficients for i=1..15
    # Encode coeffs as base-3 integer for compact storage
    # c_i in {-1, 0, 1} -> map to 0, 1, 2
    # val = c1 * 3^0 + c2 * 3^1 + ...
    
    def generate_sums(start_idx, end_idx):
        # start_idx, end_idx are 1-based indices for 1/i
        sums = [(0, 0)] # (sum_value, encoded_coeffs)
        power = 1
        for i in range(start_idx, end_idx + 1):
            k = D // i
            new_sums = []
            # c = -1 -> code 0? 
            # Let's use:
            # code 0: c = -1
            # code 1: c = 0
            # code 2: c = 1
            # sum_change = c * k
            # -1 * k = -k
            # 0 * k = 0
            # 1 * k = k
            
            for s, code in sums:
                # c = -1
                new_code = code | (0 * power) # 0
                new_sums.append((s - k, new_code))
                # c = 0
                new_code = code | (1 * power)
                new_sums.append((s, new_code))
                # c = 1
                new_code = code | (2 * power)
                new_sums.append((s + k, new_code))
            sums = new_sums
            power *= 3
            
        return sums

    # Generate Set A (1..15) and Set B (16..30)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
        
    sums_A = generate_sums(1, 15)
    sums_B = generate_sums(16, 30)
    
    # Sort by sum value
    sums_A.sort(key=lambda x: x[0])
    sums_B.sort(key=lambda x: x[0])
    
    n_A = len(sums_A)
    n_B = len(sums_B)
    
    best_abs = float('inf')
    best_sum_val = 0
    best_code_A = 0
    best_code_B = 0
    
    i = 0
    j = n_B - 1
    
    while i < n_A and j >= 0:
        if time.time() > deadline:
            break
            
        s_A, code_A = sums_A[i]
        s_B, code_B = sums_B[j]
        total_sum = s_A + s_B
        
        abs_sum = abs(total_sum)
        
        if abs_sum < best_abs and total_sum != 0:
            best_abs = abs_sum
            best_sum_val = total_sum
            best_code_A = code_A
            best_code_B = code_B
            
            if best_abs == 1:
                # Cannot get better than 1 for integer numerator
                break
                
        if total_sum > 0:
            j -= 1
        elif total_sum < 0:
            i += 1
        else:
            # total_sum == 0
            # Check neighbors to find non-zero
            if i < n_A - 1:
                s_A_next, code_A_next = sums_A[i + 1]
                cand = s_A_next + s_B
                if abs(cand) < best_abs and cand != 0:
                    best_abs = abs(cand)
                    best_sum_val = cand
                    best_code_A = code_A_next
                    best_code_B = code_B
            if j > 0:
                s_B_prev, code_B_prev = sums_B[j - 1]
                cand = s_A + s_B_prev
                if abs(cand) < best_abs and cand != 0:
                    best_abs = abs(cand)
                    best_sum_val = cand
                    best_code_A = code_A
                    best_code_B = code_B_prev
            
            # Move pointers to continue
            i += 1
            j -= 1

    # Reconstruct coefficients
    # If no non-zero sum found (unlikely), fallback
    if best_abs == float('inf'):
        return [0] * 28 + [1, -1]
        
    coeffs = [0] * 30
    
    # Decode A (indices 0..14 correspond to i=1..15)
    code = best_code_A
    power = 1
    for idx in range(15):
        digit = (code // power) % 3
        # 0 -> -1, 1 -> 0, 2 -> 1
        if digit == 0:
            coeffs[idx] = -1
        elif digit == 1:
            coeffs[idx] = 0
        else:
            coeffs[idx] = 1
        power *= 3
        
    # Decode B (indices 15..29 correspond to i=16..30)
    code = best_code_B
    power = 1
    for idx in range(15):
        digit = (code // power) % 3
        if digit == 0:
            coeffs[15 + idx] = -1
        elif digit == 1:
            coeffs[15 + idx] = 0
        else:
            coeffs[15 + idx] = 1
        power *= 3
        
    return coeffs
