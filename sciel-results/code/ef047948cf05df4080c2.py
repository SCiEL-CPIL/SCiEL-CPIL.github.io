"""Strategy: Dynamic Programming over LCM Sums with Bitset Optimization."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    n = 30
    # We want to minimize |sum c_i / i| != 0
    # c_i in {-1, 0, 1}
    
    # LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    # Precompute weights
    weights = [L // (i + 1) for i in range(n)]
    
    # Dynamic Programming
    # State: set of achievable sums after processing first i terms
    # We use a set of integers.
    # To keep it efficient, we can prune sums that are "too large" relative to the current best?
    # Actually, the number of reachable sums is bounded by the number of subsets, which is 3^30 ~ 2e14.
    # This is too large for a simple set if we don't prune.
    # However, the range of sums is [-L, L]. L ~ 2.3e12.
    # This is also too large for a boolean array.
    # But the number of *distinct* sums is much smaller than 3^30?
    # Let's estimate. The weights are large.
    # Maybe we can use a dictionary mapping sum -> last index used? No, we just need reachability.
    
    # Let's try a different approach: Meet in the Middle.
    # Split n=30 into two halves of 15.
    # Generate all 3^15 ~ 14 million sums for the first half.
    # Generate all 3^15 ~ 14 million sums for the second half.
    # Sort the second half sums.
    # For each sum in the first half, binary search in the second half for the sum closest to -sum1.
    # This is O(3^(n/2) * n) which is very feasible.
    
    # Let's implement Meet in the Middle.
    
    n1 = n // 2
    n2 = n - n1
    
    # Generate sums for first half
    sums1 = [0]
    for i in range(n1):
        w = weights[i]
        new_sums = []
        for s in sums1:
            new_sums.append(s)
            new_sums.append(s + w)
            new_sums.append(s - w)
        sums1 = new_sums
        
    # Generate sums for second half
    sums2 = [0]
    for i in range(n2):
        w = weights[n1 + i]
        new_sums = []
        for s in sums2:
            new_sums.append(s)
            new_sums.append(s + w)
            new_sums.append(s - w)
        sums2 = new_sums
        
    # Sort sums2
    sums2.sort()
    
    best_val = None
    best_coeffs = None
    
    # For each sum in sums1, find the best matching sum in sums2
    # We want to minimize |s1 + s2| subject to s1 + s2 != 0.
    
    # To recover the coefficients, we need to store the path or reconstruct it.
    # Storing the full vector for 14 million sums is too memory intensive (14M * 30 bytes ~ 420MB).
    # Instead, we can store the "choice" at each step, or use a dictionary from sum to one representative vector?
    # No, multiple vectors can yield the same sum. We just need ONE vector that yields the minimal non-zero sum.
    
    # Let's store the sum and the index of the vector in a list?
    # Or, we can store the sum and the "state" (which choices were made).
    # Since 3^15 is 14,348,907, we can store the sum and a 15-digit base-3 number representing the choices.
    # 15 digits base 3 fits in an integer (3^15 ~ 1.4e7 < 2^24).
    
    # Let's generate (sum, code) pairs.
    pairs1 = [(0, 0)]
    for i in range(n1):
        w = weights[i]
        new_pairs = []
        for s, code in pairs1:
            # Choice 0: coeff 0
            new_pairs.append((s, code))
            # Choice 1: coeff 1
            new_pairs.append((s + w, code + (1 << (2 * i))))
            # Choice 2: coeff -1
            new_pairs.append((s - w, code + (2 << (2 * i))))
        pairs1 = new_pairs
        
    pairs2 = [(0, 0)]
    for i in range(n2):
        w = weights[n1 + i]
        new_pairs = []
        for s, code in pairs2:
            # Choice 0: coeff 0
            new_pairs.append((s, code))
            # Choice 1: coeff 1
            new_pairs.append((s + w, code + (1 << (2 * i))))
            # Choice 2: coeff -1
            new_pairs.append((s - w, code + (2 << (2 * i))))
        pairs2 = new_pairs
        
    # Sort pairs2 by sum
    pairs2.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    sums2_only = [p[0] for p in pairs2]
    
    # For each pair in pairs1, find the best match in pairs2
    import bisect
    
    for s1, code1 in pairs1:
        # We want s2 such that s1 + s2 is close to 0 but not 0.
        # Target s2 = -s1.
        # Find insertion point of -s1 in sums2_only.
        target = -s1
        pos = bisect.bisect_left(sums2_only, target)
        
        # Check candidates around pos
        candidates = []
        if pos < len(sums2_only):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        # Also check pos+1 and pos-2 just in case?
        # Bisect_left gives the first element >= target.
        # So pos is the first >= target. pos-1 is the last < target.
        # These are the two closest.
        
        for idx in candidates:
            if 0 <= idx < len(pairs2):
                s2, code2 = pairs2[idx]
                total_sum = s1 + s2
                if total_sum == 0:
                    continue
                abs_sum = abs(total_sum)
                
                if best_val is None or abs_sum < best_val:
                    best_val = abs_sum
                    # Reconstruct coefficients
                    coeffs = [0] * n
                    # Decode code1
                    c = code1
                    for i in range(n1):
                        val = (c >> (2 * i)) & 3
                        if val == 1:
                            coeffs[i] = 1
                        elif val == 2:
                            coeffs[i] = -1
                    # Decode code2
                    c = code2
                    for i in range(n2):
                        val = (c >> (2 * i)) & 3
                        if val == 1:
                            coeffs[n1 + i] = 1
                        elif val == 2:
                            coeffs[n1 + i] = -1
                    best_coeffs = coeffs
                    # Check time
                    if time.time() > deadline:
                        break
        
        if time.time() > deadline:
            break
        
    # If no valid result found (should not happen), return seed
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
        
    return best_coeffs
# EVOLVE-BLOCK-END
