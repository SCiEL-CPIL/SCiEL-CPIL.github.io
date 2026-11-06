import math
import bisect

def construct():
    N = 30
    # Calculate LCM of 1 to 30
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    
    weights = [L // i for i in range(1, 31)]
    
    # Split indices into two halves
    n1 = N // 2
    n2 = N - n1
    
    indices1 = list(range(n1))
    indices2 = list(range(n1, N))
    
    # Helper to generate all sums and encoded coefficients for a set of indices
    # Coefficients are in {-1, 0, 1}
    # Encoding: base 3. -1 -> 0, 0 -> 1, 1 -> 2? 
    # Or just store the tuple? 
    # Let's use base 3 encoding for memory efficiency.
    # val -1 -> 0, 0 -> 1, 1 -> 2
    
    def generate_sums(indices):
        sums = []
        encs = []
        # Use iterative generation to avoid recursion depth issues
        # Start with [0]
        # For each index, expand each existing sum by adding -w, 0, +w
        
        # Initial state: one sum 0, one encoding 0 (all zeros)
        # Encoding logic: 
        # We have n indices. 
        # Enc is an integer. 
        # Digit j (0 to n-1) corresponds to index j in the local list.
        # Digit value: 0 for -1, 1 for 0, 2 for 1.
        
        current_sums = [0]
        current_encs = [0]
        
        for k, idx in enumerate(indices):
            w = weights[idx]
            new_sums = []
            new_encs = []
            # For each existing sum, create 3 new ones
            # We need to shift the encoding to make room for the new digit at position k
            # Current encs have k digits (positions 0 to k-1).
            # New digit is at position k.
            # So new_enc = old_enc + digit * (3**k)
            pow3 = 3 ** k
            
            for s, e in zip(current_sums, current_encs):
                # c = -1
                new_sums.append(s - w)
                new_encs.append(e + 0 * pow3)
                
                # c = 0
                new_sums.append(s)
                new_encs.append(e + 1 * pow3)
                
                # c = 1
                new_sums.append(s + w)
                new_encs.append(e + 2 * pow3)
            
            current_sums = new_sums
            current_encs = new_encs
            
        return current_sums, current_encs

    # Generate for half 1
    sums1, encs1 = generate_sums(indices1)
    
    # Sort by sum
    # Create list of tuples (sum, enc)
    pairs = list(zip(sums1, encs1))
    pairs.sort(key=lambda x: x[0])
    
    sorted_sums = [p[0] for p in pairs]
    sorted_encs = [p[1] for p in pairs]
    
    best_val = float('inf')
    best_enc1 = 0
    best_enc2 = 0
    
    # Generate for half 2 and search
    sums2, encs2 = generate_sums(indices2)
    
    for s2, e2 in zip(sums2, encs2):
        target = -s2
        # Find closest sum in sorted_sums to target
        idx = bisect.bisect_left(sorted_sums, target)
        
        candidates = []
        if idx < len(sorted_sums):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        for i in candidates:
            s1 = sorted_sums[i]
            total_sum = s1 + s2
            if total_sum == 0:
                continue
            abs_sum = abs(total_sum)
            if abs_sum < best_val:
                best_val = abs_sum
                best_enc1 = sorted_encs[i]
                best_enc2 = e2
                if best_val == 1:
                    # Decode and return
                    # We found the minimum possible non-zero value
                    # Break early? 
                    # We need to decode best_enc1 and best_enc2
                    # But we are in the loop. Let's set a flag or just break out.
                    # Since 1 is the global minimum, we can stop.
                    pass # We'll break after the loop or use a flag

    # If we didn't break early, we have the best_val.
    # But we need to construct the final coefficients.
    
    # Decode best_enc1 for indices1
    c = [0] * N
    enc = best_enc1
    for k, idx in enumerate(indices1):
        digit = enc % 3
        enc //= 3
        if digit == 0:
            c[idx] = -1
        elif digit == 1:
            c[idx] = 0
        else:
            c[idx] = 1
            
    # Decode best_enc2 for indices2
    enc = best_enc2
    for k, idx in enumerate(indices2):
        digit = enc % 3
        enc //= 3
        if digit == 0:
            c[idx] = -1
        elif digit == 1:
            c[idx] = 0
        else:
            c[idx] = 1
            
    return c
