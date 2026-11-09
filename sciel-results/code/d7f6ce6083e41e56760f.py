"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import bisect

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
            sums = [s - val for s in sums] + sums + [s + val for s in sums]
        return sums

    # Split indices 1..30 into two halves of 15
    half1 = list(range(1, 16))
    half2 = list(range(16, 31))
    
    # Generate sums for half 1
    sums1 = generate_sums_only(half1)
    
    if time.time() > deadline - 20:
        return [0] * 28 + [1, -1]
        
    sums1.sort()
    
    # Generate sums for half 2
    sums2 = generate_sums_only(half2)
    
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]
        
    # Find best pair (s1, s2) such that |s1 + s2| is minimized and non-zero
    best_found_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    for s2 in sums2:
        if time.time() > deadline - 5:
            break
            
        target = -s2
        pos = bisect.bisect_left(sums1, target)
        
        candidates = []
        if pos < len(sums1):
            candidates.append(sums1[pos])
        if pos > 0:
            candidates.append(sums1[pos-1])
            
        for s1 in candidates:
            total = s1 + s2
            if total == 0:
                continue
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
    
    # Local search polish: try flipping individual coefficients to improve
    if time.time() < deadline - 3:
        def calc_scaled_sum(coeffs):
            return sum(c * (L // (i + 1)) for i, c in enumerate(coeffs))
        
        current_sum = calc_scaled_sum(final_coeffs)
        current_abs = abs(current_sum)
        
        # Try single flips
        for i in range(n):
            if time.time() > deadline - 3:
                break
            old_c = final_coeffs[i]
            # Try changing to other values
            for new_c in [-1, 0, 1]:
                if new_c == old_c:
                    continue
                final_coeffs[i] = new_c
                new_sum = calc_scaled_sum(final_coeffs)
                if new_sum != 0:
                    new_abs = abs(new_sum)
                    if new_abs < current_abs:
                        current_abs = new_abs
                        current_sum = new_sum
                    else:
                        final_coeffs[i] = old_c
                else:
                    final_coeffs[i] = old_c
        
        # Try double flips
        if time.time() < deadline - 2:
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 2:
                        break
                    old_i = final_coeffs[i]
                    old_j = final_coeffs[j]
                    for new_i in [-1, 0, 1]:
                        if new_i == old_i:
                            continue
                        for new_j in [-1, 0, 1]:
                            if new_j == old_j:
                                continue
                            final_coeffs[i] = new_i
                            final_coeffs[j] = new_j
                            new_sum = calc_scaled_sum(final_coeffs)
                            if new_sum != 0:
                                new_abs = abs(new_sum)
                                if new_abs < current_abs:
                                    current_abs = new_abs
                                    current_sum = new_sum
                                else:
                                    final_coeffs[i] = old_i
                                    final_coeffs[j] = old_j
                            else:
                                final_coeffs[i] = old_i
                                final_coeffs[j] = old_j

    # Verify
    total = sum(final_coeffs[i] * (L // (i + 1)) for i in range(n))
    if total == 0:
        if final_coeffs[-1] == 0:
            final_coeffs[-1] = 1
        else:
            final_coeffs[-1] = -final_coeffs[-1]
            
    return final_coeffs

# EVOLVE-BLOCK-END
