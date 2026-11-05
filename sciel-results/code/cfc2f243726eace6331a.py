import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    n = 30
    
    # Calculate L = lcm(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # A_i = L / i
    A = [L // i for i in range(1, n + 1)]
    
    # Meet-in-the-middle approach:
    # Split indices into two halves of 15 each
    # For each half, enumerate all 3^15 = 14348907 possible coefficient assignments
    # Compute the integer sum for each, store in a sorted structure
    # Then find the pair of sums from the two halves whose combined sum is closest to 0 but nonzero
    
    half1 = list(range(15))  # indices 0-14
    half2 = list(range(15, 30))  # indices 15-29
    
    # Precompute A values for each half
    A1 = [A[i] for i in half1]
    A2 = [A[i] for i in half2]
    
    # Enumerate all 3^15 combinations for first half
    # Store (sum, coeffs) in a list, then sort by sum
    sums1 = []
    coeffs1 = []
    
    # Use iterative approach to avoid recursion depth issues
    # For 3^15, we can use a base-3 counter
    
    count = 3 ** 15
    for k in range(count):
        if time.time() - start_time > deadline - 10:
            break
        # Convert k to base-3 with 15 digits
        digits = []
        temp = k
        for _ in range(15):
            digits.append(temp % 3)
            temp //= 3
        # Map 0->-1, 1->0, 2->1
        coeffs = [d - 1 for d in digits]
        s = 0
        for i in range(15):
            if coeffs[i] != 0:
                s += coeffs[i] * A1[i]
        sums1.append(s)
        coeffs1.append(coeffs)
    
    # Sort by sum
    order1 = sorted(range(len(sums1)), key=lambda i: sums1[i])
    
    # Enumerate all 3^15 combinations for second half
    sums2 = []
    coeffs2 = []
    
    for k in range(count):
        if time.time() - start_time > deadline - 5:
            break
        digits = []
        temp = k
        for _ in range(15):
            digits.append(temp % 3)
            temp //= 3
        coeffs = [d - 1 for d in digits]
        s = 0
        for i in range(15):
            if coeffs[i] != 0:
                s += coeffs[i] * A2[i]
        sums2.append(s)
        coeffs2.append(coeffs)
    
    # Sort sums2 for binary search
    order2 = sorted(range(len(sums2)), key=lambda i: sums2[i])
    sorted_sums2 = [sums2[order2[i]] for i in range(len(sums2))]
    
    # For each sum in first half, find the best matching sum in second half
    # such that total sum is nonzero and |total| is minimized
    
    best_abs = float('inf')
    best_total = None
    best_c1 = None
    best_c2 = None
    
    import bisect
    
    for idx1 in range(len(sums1)):
        s1 = sums1[idx1]
        # We want s1 + s2 to be close to 0 but not 0
        # So we want s2 close to -s1
        target = -s1
        
        # Binary search for target in sorted_sums2
        pos = bisect.bisect_left(sorted_sums2, target)
        
        # Check candidates around pos
        for delta in range(-2, 3):
            j = pos + delta
            if 0 <= j < len(sorted_sums2):
                s2 = sorted_sums2[j]
                total = s1 + s2
                if total != 0:
                    abs_total = abs(total)
                    if abs_total < best_abs:
                        best_abs = abs_total
                        best_total = total
                        best_c1 = coeffs1[idx1][:]
                        best_c2 = coeffs2[order2[j]][:]
    
    # Construct final coefficients
    if best_c1 is not None and best_c2 is not None:
        result = [0] * n
        for i in range(15):
            result[i] = best_c1[i]
        for i in range(15):
            result[15 + i] = best_c2[i]
        
        # Verify the sum is nonzero
        final_sum = 0
        for i in range(n):
            if result[i] != 0:
                final_sum += result[i] * A[i]
        
        if final_sum != 0:
            return result
    
    # Fallback: simple valid solution
    result = [0] * n
    result[28] = 1
    result[29] = -1
    return result
