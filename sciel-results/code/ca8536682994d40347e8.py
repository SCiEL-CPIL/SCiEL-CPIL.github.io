"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import bisect
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    def generate_sums_with_coeffs(indices):
        """
        Generates all sums for the given indices.
        Returns a list of (sum, coeffs_tuple).
        This is only used for small sets (<= 10 terms).
        """
        sums = [(0, ())]
        for idx in indices:
            val = L // idx
            new_sums = []
            for s, c in sums:
                new_sums.append((s - val, c + (-1,)))
                new_sums.append((s, c + (0,)))
                new_sums.append((s + val, c + (1,)))
            sums = new_sums
        return sums

    def generate_sums_only(indices):
        """
        Generates all sums for the given indices.
        Returns a list of sums.
        Used for larger sets (15 terms).
        """
        sums = [0]
        for idx in indices:
            val = L // idx
            # Use array for memory efficiency during generation
            current = array('q', sums)
            new_sums = array('q', [0]) * (len(current) * 3)
            offset = 0
            for s in current:
                new_sums[offset] = s - val
                new_sums[offset + 1] = s
                new_sums[offset + 2] = s + val
                offset += 3
            sums = new_sums
        return sums

    # Split indices 1..30 into two halves of 15
    half1 = list(range(1, 16))
    half2 = list(range(16, 31))
    
    # Generate sums for half 1
    sums1 = generate_sums_only(half1)
    
    if time.time() > deadline - 20:
        return [0] * 28 + [1, -1]
        
    sums1 = array('q', sums1)
    sums1 = sorted(sums1)
    sums1 = array('q', sums1)
    
    # Generate sums for half 2
    sums2 = generate_sums_only(half2)
    
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]
    
    # Find best pair (s1, s2) such that |s1 + s2| is minimized and non-zero
    best_found_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    # Get unique values from sums2 to reduce iterations
    # sums2 has 3^15 = 14,348,907 elements
    # Many will be duplicates
    unique_sums2 = set(sums2)
    
    # Sort unique sums2 for deterministic behavior and early termination potential
    unique_sums2_sorted = sorted(unique_sums2)
    
    # Iterate over unique sums2
    for s2 in unique_sums2_sorted:
        if time.time() > deadline - 5:
            break
            
        target = -s2
        pos = bisect.bisect_left(sums1, target)
        
        # Check pos and pos-1
        if pos < len(sums1):
            s1 = sums1[pos]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_found_val:
                    best_found_val = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    if best_found_val == 1:
                        break
        
        if pos > 0:
            s1 = sums1[pos-1]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_found_val:
                    best_found_val = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    if best_found_val == 1:
                        break
        
        if best_found_val == 1:
            break
            
    if best_s1 is None:
        return [0] * 28 + [1, -1]

    # Now reconstruct coefficients for best_s1 and best_s2
    def find_coeffs_for_target(target, indices):
        """
        Finds coefficients for the given indices that sum to target.
        Uses MITM on the indices themselves.
        """
        mid = len(indices) // 2
        left_idx = indices[:mid]
        right_idx = indices[mid:]
        
        left_sums = generate_sums_with_coeffs(left_idx)
        right_sums = generate_sums_with_coeffs(right_idx)
        
        right_dict = {}
        for s, c in right_sums:
            if s not in right_dict:
                right_dict[s] = c
                
        for s_l, c_l in left_sums:
            needed = target - s_l
            if needed in right_dict:
                c_r = right_dict[needed]
                return c_l + c_r
                
        return None

    coeffs1 = find_coeffs_for_target(best_s1, half1)
    if coeffs1 is None:
        return [0] * 28 + [1, -1]
        
    coeffs2 = find_coeffs_for_target(best_s2, half2)
    if coeffs2 is None:
        return [0] * 28 + [1, -1]
        
    final_coeffs = list(coeffs1) + list(coeffs2)
    
    # Local search polish: try flipping individual coefficients to reduce |sum|
    current_sum = sum(final_coeffs[i] * (L // (i + 1)) for i in range(n))
    
    # Greedy local search: try each coefficient change
    for _ in range(100):  # Up to 100 passes
        if time.time() > deadline - 3:
            break
        improved = False
        for i in range(n):
            old_c = final_coeffs[i]
            val = L // (i + 1)
            for new_c in (-1, 0, 1):
                if new_c == old_c:
                    continue
                new_sum = current_sum + (new_c - old_c) * val
                if new_sum == 0:
                    continue
                if abs(new_sum) < abs(current_sum):
                    final_coeffs[i] = new_c
                    current_sum = new_sum
                    improved = True
                    break
            if time.time() > deadline - 3:
                break
        if not improved:
            break
    
    # Final verification
    total = sum(final_coeffs[i] * (L // (i + 1)) for i in range(n))
    if total == 0:
        if final_coeffs[-1] == 0:
            final_coeffs[-1] = 1
        else:
            final_coeffs[-1] = -final_coeffs[-1]
            
    return final_coeffs

# EVOLVE-BLOCK-END
