"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import itertools

def construct():
    # Compute L = lcm(1, 2, ..., 30)
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    
    # w[i] = L / (i+1) for i in 0..29
    w = [L // (i + 1) for i in range(30)]
    
    n1 = 15
    n2 = 15
    
    # Generate all (sum, state) for the first half (indices 0..14)
    # state uses 2 bits per index.
    # Global index j corresponds to bits 2*j and 2*j+1.
    # 00: 0, 01: +1, 10: -1
    
    list1 = [(0, 0)]
    for idx in range(n1):
        w_idx = w[idx]
        new_list1 = []
        for s, st in list1:
            # 0
            new_list1.append((s, st))
            # +1
            new_list1.append((s + w_idx, st | (1 << (2 * idx))))
            # -1
            new_list1.append((s - w_idx, st | (2 << (2 * idx))))
        list1 = new_list1
        
    # Generate all (sum, state) for the second half (indices 15..29)
    list2 = [(0, 0)]
    for idx in range(n2):
        w_idx = w[15 + idx]
        new_list2 = []
        for s, st in list2:
            # 0
            new_list2.append((s, st))
            # +1
            new_list2.append((s + w_idx, st | (1 << (2 * (15 + idx)))))
            # -1
            new_list2.append((s - w_idx, st | (2 << (2 * (15 + idx)))))
        list2 = new_list2
        
    # Sort list1 by sum
    list1.sort(key=lambda x: x[0])
    
    # For list2, we want to find s2 such that s1 + s2 is small.
    # This is equivalent to finding s2 close to -s1.
    # Let's create a list of (-s2, state2) and sort it.
    list2_neg = [(-s, st) for s, st in list2]
    list2_neg.sort(key=lambda x: x[0])
    
    # Two-pointer scan to find min |s1 - (-s2)| != 0
    # s1 from list1, t2 from list2_neg (where t2 = -s2)
    # We want min |s1 - t2| such that s1 != t2
    
    i = 0
    j = 0
    n1_size = len(list1)
    n2_size = len(list2_neg)
    
    best_abs_T = float('inf')
    best_i = -1
    best_j = -1
    
    while i < n1_size and j < n2_size:
        s1, st1 = list1[i]
        t2, st2 = list2_neg[j]
        
        diff = s1 - t2
        abs_diff = abs(diff)
        
        if diff != 0:
            if abs_diff < best_abs_T:
                best_abs_T = abs_diff
                best_i = i
                best_j = j
                if best_abs_T == 1:
                    break
        else:
            # diff == 0, sum is 0. We need non-zero.
            # Move both pointers to find next candidate?
            # Or just move one?
            # If we move i, we might miss a smaller diff with the same j and i+1?
            # If s1 == t2, then s1 < t2_next and s1_next > t2.
            # The next smallest diffs are |s1_next - t2| and |s1 - t2_next|.
            # So we can move both, or just one and let the loop handle it.
            # Moving both is efficient.
            pass 
        
        if s1 < t2:
            i += 1
        elif s1 > t2:
            j += 1
        else:
            # s1 == t2
            i += 1
            j += 1
            
    # Reconstruct coefficients
    if best_i == -1:
        # Fallback, shouldn't happen
        return [0]*30
        
    st1 = list1[best_i][1]
    st2 = list2_neg[best_j][1]
    full_state = st1 | st2
    
    coeffs = [0] * 30
    for i in range(30):
        val = (full_state >> (2 * i)) & 3
        if val == 1:
            coeffs[i] = 1
        elif val == 2:
            coeffs[i] = -1
        else:
            coeffs[i] = 0
            
    return coeffs
# EVOLVE-BLOCK-END
