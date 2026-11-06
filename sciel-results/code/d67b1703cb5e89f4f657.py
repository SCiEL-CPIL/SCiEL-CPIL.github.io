"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 30
    mid = N // 2  # 15

    def lcm(a, b):
        return a * b // gcd(a, b)

    # LCM(1..30)
    L = reduce(lcm, range(1, N + 1))

    # weights[i] = L // (i+1), so c[i]/(i+1) = c[i]*weights[i] / L
    weights = np.array([L // (i + 1) for i in range(N)], dtype=np.int64)

    w1 = weights[:mid]  # denominators 1..15
    w2 = weights[mid:]  # denominators 16..30

    def generate_sums_and_codes(weights_half):
        """
        Generate all 3^n sums and their base-3 codes.
        Code encoding: digit 0 -> -1, digit 1 -> 0, digit 2 -> +1.
        The first weight processed becomes the most significant digit.
        """
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)

        for w in weights_half:
            # Sums
            s_minus = sums - w
            s_zero = sums
            s_plus = sums + w

            # Codes: append new digit (least significant of the new code)
            # digit 0 -> -1, digit 1 -> 0, digit 2 -> +1
            c_minus = codes * 3      # digit 0
            c_zero = codes * 3 + 1   # digit 1
            c_plus = codes * 3 + 2   # digit 2

            sums = np.concatenate([s_minus, s_zero, s_plus])
            codes = np.concatenate([c_minus, c_zero, c_plus])

        return sums, codes

    def decode_code(code, n):
        """Decode a base-3 code into n coefficients in {-1, 0, 1}.
        The most significant digit corresponds to the first weight."""
        coeffs = [0] * n
        # Extract digits from least significant to most significant
        for i in range(n - 1, -1, -1):
            d = code % 3
            if d == 0:
                coeffs[i] = -1
            elif d == 1:
                coeffs[i] = 0
            else:
                coeffs[i] = 1
            code //= 3
        return coeffs

    # Check time
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Generate sums and codes for half 1
    sums1, codes1 = generate_sums_and_codes(w1)

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Sort sums1 and align codes
    order1 = np.argsort(sums1, kind='stable')
    sums1_sorted = sums1[order1]
    codes1_sorted = codes1[order1]

    # Generate sums and codes for half 2
    sums2, codes2 = generate_sums_and_codes(w2)

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # For each s2 in sums2, find s1 in sums1_sorted such that s1 + s2 is closest to 0 but nonzero.
    # Target s1 = -s2.
    targets = -sums2
    len_s1 = len(sums1_sorted)

    # Predecessor: largest index with sums1_sorted[idx] < target
    idx_l = np.searchsorted(sums1_sorted, targets, side='left')
    
    # Candidate 1: predecessor (idx_l - 1)
    idx_cand1 = idx_l - 1
    valid1 = idx_cand1 >= 0
    idx_cand1_clipped = np.clip(idx_cand1, 0, len_s1 - 1)
    s1_cand1 = sums1_sorted[idx_cand1_clipped]
    code1_cand1 = codes1_sorted[idx_cand1_clipped]

    total1 = sums2 + s1_cand1
    abs_total1 = np.abs(total1)

    # Mask out invalid (idx < 0) and zero sums
    mask1 = valid1 & (total1 != 0)
    abs_total1[~mask1] = np.iinfo(np.int64).max

    # Candidate 2: successor (idx_l)
    idx_cand2 = idx_l
    valid2 = idx_cand2 < len_s1
    idx_cand2_clipped = np.clip(idx_cand2, 0, len_s1 - 1)
    s1_cand2 = sums1_sorted[idx_cand2_clipped]
    code1_cand2 = codes1_sorted[idx_cand2_clipped]

    total2 = sums2 + s1_cand2
    abs_total2 = np.abs(total2)

    # Mask out invalid (idx >= len) and zero sums
    mask2 = valid2 & (total2 != 0)
    abs_total2[~mask2] = np.iinfo(np.int64).max

    # Find global minimum
    min_abs1 = np.min(abs_total1)
    min_abs2 = np.min(abs_total2)

    if min_abs1 <= min_abs2:
        best_abs = min_abs1
        idx_best = np.argmin(abs_total1)
        s2_val = sums2[idx_best]
        s1_val = s1_cand1[idx_best]
        code1_val = code1_cand1[idx_best]
        code2_val = codes2[idx_best]
    else:
        best_abs = min_abs2
        idx_best = np.argmin(abs_total2)
        s2_val = sums2[idx_best]
        s1_val = s1_cand2[idx_best]
        code1_val = code1_cand2[idx_best]
        code2_val = codes2[idx_best]

    if best_abs == np.iinfo(np.int64).max:
        # No valid solution found (shouldn't happen), fallback
        return [0] * 28 + [1, -1]

    # Decode coefficients
    coeffs1 = decode_code(code1_val, mid)
    coeffs2 = decode_code(code2_val, mid)

    result = coeffs1 + coeffs2

    # Verify
    total = 0
    for i in range(N):
        total += result[i] * weights[i]
    if total == 0:
        # Fallback (shouldn't happen due to masking)
        return [0] * 28 + [1, -1]

    # Check if we can improve using modular arithmetic approach
    if time.time() < deadline - 5.0:
        # Try modular approach with a prime P > 3^15
        # 3^15 = 14348907, so P should be > 14348907
        # Use a prime around 20 million for good balance
        P = 20000003  # Check if prime
        
        # Verify P is prime (simple trial division for small range)
        def is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        if not is_prime(P):
            # Find next prime
            P += 2
            while not is_prime(P):
                P += 2
        
        # Reduce weights modulo P
        w1_mod = np.mod(w1, P).astype(np.int64)
        w2_mod = np.mod(w2, P).astype(np.int64)
        
        # Generate modular sums for half 1
        sums1_mod = np.array([0], dtype=np.int64)
        codes1_mod = np.array([0], dtype=np.int64)
        for w in w1_mod:
            s_minus = (sums1_mod - w) % P
            s_zero = sums1_mod
            s_plus = (sums1_mod + w) % P
            
            c_minus = codes1_mod * 3
            c_zero = codes1_mod * 3 + 1
            c_plus = codes1_mod * 3 + 2
            
            sums1_mod = np.concatenate([s_minus, s_zero, s_plus])
            codes1_mod = np.concatenate([c_minus, c_zero, c_plus])
        
        if time.time() > deadline:
            return [int(c) for c in result]
        
        # Generate modular sums for half 2
        sums2_mod = np.array([0], dtype=np.int64)
        codes2_mod = np.array([0], dtype=np.int64)
        for w in w2_mod:
            s_minus = (sums2_mod - w) % P
            s_zero = sums2_mod
            s_plus = (sums2_mod + w) % P
            
            c_minus = codes2_mod * 3
            c_zero = codes2_mod * 3 + 1
            c_plus = codes2_mod * 3 + 2
            
            sums2_mod = np.concatenate([s_minus, s_zero, s_plus])
            codes2_mod = np.concatenate([c_minus, c_zero, c_plus])
        
        if time.time() > deadline:
            return [int(c) for c in result]
        
        # Sort sums1_mod
        order_mod = np.argsort(sums1_mod, kind='stable')
        sums1_mod_sorted = sums1_mod[order_mod]
        codes1_mod_sorted = codes1_mod[order_mod]
        
        # For each s2_mod, find s1_mod such that (s1_mod + s2_mod) % P is close to 0 but not 0
        # We want (s1_mod + s2_mod) % P to be as small as possible but nonzero
        # This means s1_mod should be close to (-s2_mod) % P
        
        targets_mod = (-sums2_mod) % P
        len_s1_mod = len(sums1_mod_sorted)
        
        # Search for closest match
        idx_l_mod = np.searchsorted(sums1_mod_sorted, targets_mod, side='left')
        
        # Candidate 1: predecessor
        idx_cand1_mod = idx_l_mod - 1
        valid1_mod = idx_cand1_mod >= 0
        idx_cand1_mod_clipped = np.clip(idx_cand1_mod, 0, len_s1_mod - 1)
        s1_cand1_mod = sums1_mod_sorted[idx_cand1_mod_clipped]
        code1_cand1_mod = codes1_mod_sorted[idx_cand1_mod_clipped]
        
        total1_mod = (sums2_mod + s1_cand1_mod) % P
        abs_total1_mod = np.where(total1_mod > P // 2, P - total1_mod, total1_mod)
        abs_total1_mod[total1_mod == 0] = P  # Mark zero sums
        
        # Candidate 2: successor
        idx_cand2_mod = idx_l_mod
        valid2_mod = idx_cand2_mod < len_s1_mod
        idx_cand2_mod_clipped = np.clip(idx_cand2_mod, 0, len_s1_mod - 1)
        s1_cand2_mod = sums1_mod_sorted[idx_cand2_mod_clipped]
        code1_cand2_mod = codes1_mod_sorted[idx_cand2_mod_clipped]
        
        total2_mod = (sums2_mod + s1_cand2_mod) % P
        abs_total2_mod = np.where(total2_mod > P // 2, P - total2_mod, total2_mod)
        abs_total2_mod[total2_mod == 0] = P  # Mark zero sums
        
        # Find global minimum in modular space
        min_abs_mod1 = np.min(abs_total1_mod)
        min_abs_mod2 = np.min(abs_total2_mod)
        
        if min_abs_mod1 < min_abs_mod2:
            idx_best_mod = np.argmin(abs_total1_mod)
            s2_val_mod = sums2_mod[idx_best_mod]
            s1_val_mod = s1_cand1_mod[idx_best_mod]
            code1_val_mod = code1_cand1_mod[idx_best_mod]
            code2_val_mod = codes2_mod[idx_best_mod]
        else:
            idx_best_mod = np.argmin(abs_total2_mod)
            s2_val_mod = sums2_mod[idx_best_mod]
            s1_val_mod = s1_cand2_mod[idx_best_mod]
            code1_val_mod = code1_cand2_mod[idx_best_mod]
            code2_val_mod = codes2_mod[idx_best_mod]
        
        # Decode modular solution
        coeffs1_mod = decode_code(code1_val_mod, mid)
        coeffs2_mod = decode_code(code2_val_mod, mid)
        result_mod = coeffs1_mod + coeffs2_mod
        
        # Verify the modular solution in exact arithmetic
        total_exact = 0
        for i in range(N):
            total_exact += result_mod[i] * weights[i]
        
        if total_exact != 0 and abs(total_exact) < best_abs:
            result = result_mod
            best_abs = abs(total_exact)

    return [int(c) for c in result]

# EVOLVE-BLOCK-END
