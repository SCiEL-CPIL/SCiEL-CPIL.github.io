# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Precompute integer weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    half1_weights = weights[:15]
    half2_weights = weights[15:]
    
    def generate_sums_with_code(w_list):
        # Returns list of (sum_val, code)
        # code is base-3 representation of coefficients
        # c_i in {-1, 0, 1} mapped to 0, 1, 2 in base 3
        # code = sum (c_i + 1) * 3^i
        sums = [(0, 0)]
        for idx, w in enumerate(w_list):
            power = 3 ** idx
            new_sums = []
            for s, c in sums:
                # c_i = -1 -> digit 0
                new_sums.append((s - w, c))
                # c_i = 0 -> digit 1
                new_sums.append((s, c + power))
                # c_i = 1 -> digit 2
                new_sums.append((s + w, c + 2 * power))
            sums = new_sums
        return sums

    def decode_code(code, n):
        coeffs = []
        for i in range(n):
            digit = (code // (3 ** i)) % 3
            coeffs.append(digit - 1)
        return coeffs

    # Check time
    if time.time() - start_time > deadline - 10:
        return [0] * 28 + [1, -1]
        
    t0 = time.time()
    # Generate sums for half 2 with codes
    sums2_data = generate_sums_with_code(half2_weights)
    
    # Sort by sum value
    sums2_data.sort(key=lambda x: x[0])
    
    # Compress: keep only unique sums, retaining one code for each
    unique_sums2 = []
    codes_for_sums2 = []
    last_sum = None
    for s, c in sums2_data:
        if s != last_sum:
            unique_sums2.append(s)
            codes_for_sums2.append(c)
            last_sum = s
            
    # Free memory
    del sums2_data
    
    # Generate sums for half 1 with codes
    sums1_data = generate_sums_with_code(half1_weights)
    
    # Create a dictionary mapping sum to code for half 1
    # If multiple codes produce the same sum, we keep the last one (or first, doesn't matter)
    dict1 = {}
    for s, c in sums1_data:
        dict1[s] = c
        
    del sums1_data
    
    t1 = time.time()
    
    # Binary search for best pair
    best_abs_diff = float('inf')
    best_s1 = 0
    best_s2 = 0
    best_code1 = 0
    best_code2 = 0
    
    # To speed up, we can iterate over sums1 (keys of dict1)
    # For each s1, find s2 in unique_sums2 such that s1+s2 is closest to 0
    # and non-zero.
    
    for s1, c1 in dict1.items():
        if time.time() - start_time > deadline - 5:
            break
            
        # We want s2 such that s1 + s2 = 0 => s2 = -s1
        target = -s1
        
        # Binary search in unique_sums2 for target
        # bisect_left returns the index where target would be inserted
        idx = bisect.bisect_left(unique_sums2, target)
        
        # Check idx and idx-1
        candidates = []
        if idx < len(unique_sums2):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        for c_idx in candidates:
            s2 = unique_sums2[c_idx]
            diff = s1 + s2
            if diff == 0:
                continue
            abs_diff = abs(diff)
            if abs_diff < best_abs_diff:
                best_abs_diff = abs_diff
                best_s1 = s1
                best_s2 = s2
                best_code1 = c1
                best_code2 = codes_for_sums2[c_idx]
                
        if best_abs_diff == 1:
            break

    if best_abs_diff == float('inf'):
        return [0] * 28 + [1, -1]
        
    # Reconstruct coefficients
    c1 = decode_code(best_code1, 15)
    c2 = decode_code(best_code2, 15)
    
    result = c1 + c2
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        return [0] * 28 + [1, -1]
        
    return result

# EVOLVE-BLOCK-END
