import time
import bisect
from itertools import product

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    # Calculate LCM of 1..30
    # LCM(1..30) = 2329089562800
    L = 2329089562800
    
    # Weights w_i = L / (i+1) for i in 0..29
    # Denominators are 1..30
    weights = [L // (i + 1) for i in range(30)]
    
    # Split into two halves of 15
    # Half 1: indices 0..14 (denominators 1..15)
    # Half 2: indices 15..29 (denominators 16..30)
    
    w1 = weights[:15]
    w2 = weights[15:]
    
    # Generate all sums for half 1
    # To optimize, we can use a recursive generation or itertools.product
    # itertools.product is implemented in C and is fast
    
    sums1 = []
    # Pre-allocate? No, list append is fast enough.
    # 3^15 = 14348907
    for coeffs in product([-1, 0, 1], repeat=15):
        s = 0
        # Unrolling loop or using sum with zip
        # sum(c*w for c, w in zip(coeffs, w1))
        # This creates a generator, might be slow.
        # Manual loop is often faster in CPython for small n if optimized, but zip+sum is C-level.
        # Let's try manual loop for speed if needed, but zip is good.
        for c, w in zip(coeffs, w1):
            if c != 0:
                s += c * w
        sums1.append(s)
        
    if time.time() > deadline:
        return [0]*28 + [1, -1] # Fallback

    sums1.sort()
    
    # Generate all sums for half 2 and find best match
    best_abs_sum = float('inf')
    best_c1 = None
    best_c2 = None
    
    for coeffs2 in product([-1, 0, 1], repeat=15):
        s2 = 0
        for c, w in zip(coeffs2, w2):
            if c != 0:
                s2 += c * w
        
        target = -s2
        
        # Binary search in sums1 for target
        # We want to find index where target would be inserted
        idx = bisect.bisect_left(sums1, target)
        
        # Check idx and idx-1
        # Also check idx+1 just in case? No, bisect_left gives first >= target.
        # The closest values are at idx-1 and idx.
        
        candidates = []
        if 0 <= idx < len(sums1):
            candidates.append(idx)
        if 0 <= idx - 1 < len(sums1):
            candidates.append(idx - 1)
            
        for i in candidates:
            s1 = sums1[i]
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_abs_sum:
                best_abs_sum = abs_total
                best_c2 = coeffs2
                # We need to recover c1.
                # We stored only sums, not the coefficients.
                # We need a mapping from sum to coefficients?
                # Or we can store the coefficients along with the sum?
                # That doubles memory.
                # Alternatively, we can re-generate c1 from the sum?
                # No, multiple coefficients can yield the same sum.
                # But we just need *a* valid set of coefficients.
                # We can store the coefficients in a dictionary {sum: coeffs}
                # But 14M entries in a dict is heavy.
                
                # Better approach:
                # Store (sum, coeffs) in the list?
                # Or, since we just need one solution, we can store the coeffs for the best sum found.
                # But we don't know which sum is best until we check all.
                
                # Let's modify the storage:
                # Store list of (sum, coeffs)
                # Then sort by sum.
                # Then when we find the best sum, we have the coeffs.
                
                # Let's restart with this approach.
                pass

    # Re-evaluating storage strategy
    # Storing 14M tuples of (int, tuple) is memory intensive but likely okay (14M * 100 bytes ~ 1.4GB? Maybe too much).
    # Python int is 28 bytes. Tuple of 15 ints is ~100 bytes.
    # 14M * 128 bytes ~ 1.8 GB. This might exceed memory limits.
    
    # Alternative:
    # Store just sums in a list.
    # Keep a dictionary {sum: coeffs} only for the best candidates?
    # Or, we can just store the coefficients that produced the minimum sum in the first half?
    # No, the minimum sum in the first half isn't necessarily the one that combines well with the second half.
    
    # Let's try to reduce memory.
    # We can store the sums in an array of 64-bit integers if they fit?
    # Max sum for half 1: sum(w1) = L * H_15 = 2329089562800 * 3.318 ~ 7.7e12.
    # This fits in 64-bit signed integer (max ~9e18).
    # So we can use NumPy array of int64.
    # But we still need to recover the coefficients.
    
    # Idea:
    # 1. Generate all (sum, coeffs) for half 1.
    # 2. Sort by sum.
    # 3. Iterate half 2.
    # 4. If we find a new best sum, we need the coeffs for the corresponding sum in half 1.
    #    We can store a dict `sum_to_coeffs` for half 1.
    #    Size of dict: 14M entries.
    #    Memory: 14M * (key int + value tuple) ~ 14M * 150 bytes ~ 2.1 GB.
    #    This is risky.
    
    # Better Idea:
    # We don't need to store ALL coeffs.
    # We can store the coeffs for the best sum found *so far* in the search?
    # No, the best sum in half 1 is fixed.
    
    # Let's check the memory limit. Usually 2-4GB is available.
    # 14M entries in a dict is likely too slow and memory heavy in Python.
    
    # Alternative:
    # Use a list of (sum, coeffs) and sort it.
    # 14M tuples.
    # Tuple overhead is high.
    
    # What if we store the coefficients as a single integer?
    # Each coeff is -1, 0, 1. Can encode as 0, 1, 2.
    # 3 bits per coeff. 15 coeffs = 45 bits. Fits in a 64-bit int.
    # So we can store (sum, encoded_coeffs).
    # List of 14M tuples of (int64, int64).
    # Memory: 14M * 16 bytes (if packed) ~ 224 MB.
    # Python list of tuples: 14M * (8+8+overhead) ~ 14M * 50 bytes ~ 700 MB.
    # This is feasible.
    
    # Let's implement this.
    
    # Reset
    sums1_encoded = []
    
    for coeffs in product([-1, 0, 1], repeat=15):
        s = 0
        code = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * w1[i]
            # Encode c: -1->0, 0->1, 1->2
            # val = c + 1
            val = c + 1
            code |= (val << (3 * i))
        sums1_encoded.append((s, code))
        
    if time.time() > deadline:
        return [0]*28 + [1, -1]

    # Sort by sum
    sums1_encoded.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    sums1_vals = [x[0] for x in sums1_encoded]
    
    best_abs_sum = float('inf')
    best_code1 = None
    best_coeffs2 = None
    
    for coeffs2 in product([-1, 0, 1], repeat=15):
        s2 = 0
        for c, w in zip(coeffs2, w2):
            if c != 0:
                s2 += c * w
        
        target = -s2
        idx = bisect.bisect_left(sums1_vals, target)
        
        candidates = []
        if 0 <= idx < len(sums1_vals):
            candidates.append(idx)
        if 0 <= idx - 1 < len(sums1_vals):
            candidates.append(idx - 1)
            
        for i in candidates:
            s1 = sums1_vals[i]
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_abs_sum:
                best_abs_sum = abs_total
                best_code1 = sums1_encoded[i][1]
                best_coeffs2 = coeffs2
                
    # Reconstruct coefficients
    coeffs1 = [0] * 15
    for i in range(15):
        val = (best_code1 >> (3 * i)) & 7
        coeffs1[i] = val - 1
        
    coeffs = coeffs1 + list(best_coeffs2)
    
    return coeffs

construct()
