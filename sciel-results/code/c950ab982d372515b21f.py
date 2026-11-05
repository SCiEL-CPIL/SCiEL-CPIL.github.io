"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import bisect

def construct():
    # Compute L = lcm(1, 2, ..., 30)
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    
    # w[i] = L / (i+1) for i in 0..29
    w = [L // (i + 1) for i in range(30)]
    
    # Generate all (sum, state) for the second half (indices 15..29) iteratively
    list2 = [(0, 0)]
    for i in range(15):
        w_val = w[15 + i]
        offset = 2 * (15 + i)
        mask1 = 1 << offset
        mask2 = 2 << offset
        
        additions = []
        append = additions.append
        for s, st in list2:
            append((s + w_val, st | mask1))
            append((s - w_val, st | mask2))
        list2 += additions
        
    # Sort list2 by sum
    list2.sort(key=lambda x: x[0])
    
    # Extract sums for bisect
    sums2 = [s for s, _ in list2]
    
    best_abs_T = float('inf')
    best_state = 0
    
    # Generate all (sum, state) for the first half (indices 0..14) iteratively
    list1 = [(0, 0)]
    for i in range(15):
        w_val = w[i]
        offset = 2 * i
        mask1 = 1 << offset
        mask2 = 2 << offset
        
        additions = []
        append = additions.append
        for s, st in list1:
            append((s + w_val, st | mask1))
            append((s - w_val, st | mask2))
        list1 += additions
        
    # Search for best match
    for s1, st1 in list1:
        target = -s1
        pos = bisect.bisect_left(sums2, target)
        
        # Check pos
        if pos < len(sums2):
            s2, st2 = list2[pos]
            total_sum = s1 + s2
            if total_sum != 0:
                abs_t = abs(total_sum)
                if abs_t < best_abs_T:
                    best_abs_T = abs_t
                    best_state = st1 | st2
                    if best_abs_T == 1:
                        break
        # Check pos-1
        if pos > 0:
            s2, st2 = list2[pos - 1]
            total_sum = s1 + s2
            if total_sum != 0:
                abs_t = abs(total_sum)
                if abs_t < best_abs_T:
                    best_abs_T = abs_t
                    best_state = st1 | st2
                    if best_abs_T == 1:
                        break
                        
    # Construct coefficients
    coeffs = [0] * 30
    for i in range(30):
        val = (best_state >> (2 * i)) & 3
        if val == 1:
            coeffs[i] = 1
        elif val == 2:
            coeffs[i] = -1
        else:
            coeffs[i] = 0
            
    return coeffs
# EVOLVE-BLOCK-END
