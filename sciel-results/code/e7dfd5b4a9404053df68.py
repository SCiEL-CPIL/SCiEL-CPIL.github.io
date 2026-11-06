import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
        
    w = [L // i for i in range(1, n + 1)]
    
    mid = n // 2
    w1 = w[:mid]
    w2 = w[mid:]
    
    def generate_sums_and_indices(weights):
        """
        Generate all possible sums and their base-3 indices.
        Returns a list of (sum_value, index) sorted by sum_value.
        """
        # Start with sum 0, index 0
        # Index represents the choice of coefficients in base 3.
        # coeff_i = (index // 3^i) % 3 - 1
        # We build this iteratively.
        # At step k, we have processed weights[0..k-1].
        # The index for the new weights will be shifted.
        
        # To avoid huge memory overhead of tuples, we can use two lists and sort together.
        # But for 15 steps, the list grows to 3^15.
        # Let's use a list of (sum, index).
        
        current = [(0, 0)]
        power = 1 # 3^0
        
        for i, wi in enumerate(weights):
            next_curr = []
            # For each existing state, we can add -wi, 0, +wi
            # The index update:
            # The new digit (for weight i) is added at position i.
            # So new_index = old_index + digit * 3^i
            # digit 0 -> coeff -1
            # digit 1 -> coeff 0
            # digit 2 -> coeff +1
            
            d0 = 0
            d1 = 1
            d2 = 2
            
            for s, idx in current:
                # Option 1: coeff -1 (digit 0)
                next_curr.append((s - wi, idx + d0 * power))
                # Option 2: coeff 0 (digit 1)
                next_curr.append((s, idx + d1 * power))
                # Option 3: coeff +1 (digit 2)
                next_curr.append((s + wi, idx + d2 * power))
            
            current = next_curr
            power *= 3
            
            # Time check
            if time.time() > deadline:
                return None

        # Sort by sum
        current.sort(key=lambda x: x[0])
        return current

    # Generate for first half
    sums1 = generate_sums_and_indices(w1)
    if sums1 is None:
        return [0] * 28 + [1, -1]
        
    # Generate for second half
    sums2 = generate_sums_and_indices(w2)
    if sums2 is None:
        return [0] * 28 + [1, -1]

    # Find best pair
    # We want min |s1 + s2| such that s1 + s2 != 0
    # sums1 and sums2 are sorted.
    # Use two pointers or binary search.
    # Since we want closest to 0, we can iterate through one and binary search in the other.
    
    best_val = float('inf')
    best_idx1 = None
    best_idx2 = None
    
    # Extract sums for binary search
    sums1_vals = [x[0] for x in sums1]
    
    for s2_val, idx2 in sums2:
        target = -s2_val
        # Binary search for target in sums1_vals
        # We want to find the element in sums1_vals closest to target.
        # bisect_left gives insertion point.
        
        # Using bisect on the list of values
        import bisect
        pos = bisect.bisect_left(sums1_vals, target)
        
        # Check pos and pos-1
        for check_pos in [pos, pos - 1]:
            if 0 <= check_pos < len(sums1_vals):
                s1_val = sums1_vals[check_pos]
                total = s1_val + s2_val
                if total != 0:
                    if abs(total) < best_val:
                        best_val = abs(total)
                        best_idx1 = sums1[check_pos][1]
                        best_idx2 = idx2
                        
    if best_idx1 is None:
        return [0] * 28 + [1, -1]
        
    # Decode coefficients
    coeffs = [0] * n
    
    # Decode idx1 for first half
    power = 1
    for i in range(len(w1)):
        digit = (best_idx1 // power) % 3
        coeffs[i] = digit - 1
        power *= 3
        
    # Decode idx2 for second half
    power = 1
    for i in range(len(w2)):
        digit = (best_idx2 // power) % 3
        coeffs[mid + i] = digit - 1
        power *= 3
        
    return coeffs
