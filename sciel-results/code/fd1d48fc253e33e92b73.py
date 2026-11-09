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

    # Split into 4 groups of 7 + 2 groups of 1
    # Groups: [1-7], [8-14], [15-21], [22-28], [29], [30]
    groups = [
        list(range(1, 8)),    # 7 terms
        list(range(8, 15)),   # 7 terms
        list(range(15, 22)),  # 7 terms
        list(range(22, 29)),  # 7 terms
        [29],                 # 1 term
        [30],                 # 1 term
    ]
    
    # Generate sums for each group
    group_sums = []
    for g in groups:
        gs = generate_sums_only(g)
        group_sums.append(gs)
        if time.time() > deadline - 40:
            return [0] * 28 + [1, -1]
    
    # Sort the first 4 groups (each has 3^7 = 2187 elements)
    for i in range(4):
        group_sums[i] = sorted(group_sums[i])
    
    # For each combination of the two single-term groups (9 combinations),
    # do a full two-pointer sweep across the 4 seven-term groups
    best_found_val = float('inf')
    best_combo = None  # (g1_idx, g2_idx, g3_idx, g4_idx, g5_c, g6_c)
    
    # Precompute the values for single terms
    val29 = L // 29
    val30 = L // 30
    
    for c5 in (-1, 0, 1):
        for c6 in (-1, 0, 1):
            if time.time() > deadline - 5:
                break
            offset = c5 * val29 + c6 * val30
            
            # Now find min |s1+s2+s3+s4+offset| != 0
            # Use two-pointer: for each pair (s1,s2) from groups 1,2,
            # find best (s3,s4) from groups 3,4
            # But that's 2187*2187 * 2187*2187 which is too large.
            # Instead: precompute all sums of groups 3+4
            # 3^7 * 3^7 = 4.7M sums for groups 3+4
            # Then for each s12 from groups 1+2 (4.7M), binary search in groups 3+4 sums
            
            # Generate all sums of groups 3+4
            sums34 = []
            for s3 in group_sums[2]:
                for s4 in group_sums[3]:
                    sums34.append(s3 + s4)
            sums34.sort()
            
            # Now for each s12 from groups 1+2, find best match in sums34
            for s1 in group_sums[0]:
                for s2 in group_sums[1]:
                    s12 = s1 + s2
                    target = -(s12 + offset)
                    pos = bisect.bisect_left(sums34, target)
                    
                    if pos < len(sums34):
                        total = s12 + offset + sums34[pos]
                        if total != 0:
                            abs_total = abs(total)
                            if abs_total < best_found_val:
                                best_found_val = abs_total
                                best_combo = (s1, s2, None, None, c5, c6, sums34[pos])
                                if best_found_val == 1:
                                    break
                    
                    if pos > 0:
                        total = s12 + offset + sums34[pos-1]
                        if total != 0:
                            abs_total = abs(total)
                            if abs_total < best_found_val:
                                best_found_val = abs_total
                                best_combo = (s1, s2, None, None, c5, c6, sums34[pos-1])
                                if best_found_val == 1:
                                    break
                    
                    if best_found_val == 1:
                        break
                if best_found_val == 1:
                    break
            if best_found_val == 1:
                break
        if best_found_val == 1:
            break
    
    if best_combo is None:
        return [0] * 28 + [1, -1]
    
    s1_best, s2_best, _, _, c5_best, c6_best, s34_best = best_combo
    
    # Reconstruct coefficients
    def find_coeffs_for_target(target, indices):
        """
        Finds coefficients for the given indices that sum to target.
        """
        if not indices:
            return ()
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
    
    # Find coeffs for group 1 (indices 1-7)
    coeffs1 = find_coeffs_for_target(s1_best, groups[0])
    if coeffs1 is None:
        return [0] * 28 + [1, -1]
    
    # Find coeffs for group 2 (indices 8-14)
    coeffs2 = find_coeffs_for_target(s2_best, groups[1])
    if coeffs2 is None:
        return [0] * 28 + [1, -1]
    
    # Find coeffs for groups 3+4 (indices 15-28) that sum to s34_best
    # Use MITM on groups 3 and 4 separately
    mid34 = 7  # group 3 has 7 terms
    left34 = groups[2]
    right34 = groups[3]
    
    left34_sums = generate_sums_with_coeffs(left34)
    right34_sums = generate_sums_with_coeffs(right34)
    
    right34_dict = {}
    for s, c in right34_sums:
        if s not in right34_dict:
            right34_dict[s] = c
    
    coeffs34 = None
    for s_l, c_l in left34_sums:
        needed = s34_best - s_l
        if needed in right34_dict:
            c_r = right34_dict[needed]
            coeffs34 = c_l + c_r
            break
    
    if coeffs34 is None:
        return [0] * 28 + [1, -1]
    
    # Assemble final coefficients
    final_coeffs = list(coeffs1) + list(coeffs2) + list(coeffs34) + [c5_best, c6_best]
    
    # Local search polish
    current_sum = sum(final_coeffs[i] * (L // (i + 1)) for i in range(n))
    
    for _ in range(100):
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
