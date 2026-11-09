import time
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 30
    
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into 5 groups of 6
    group_size = 6
    num_groups = 5
    
    # Generate sums for each group
    # Group k covers indices k*6 to (k+1)*6 - 1
    groups = []
    for k in range(num_groups):
        start_idx = k * group_size
        end_idx = start_idx + group_size
        group_weights = weights[start_idx:end_idx]
        
        group_sums = []
        # Use itertools.product for -1, 0, 1
        for coeffs in itertools.product([-1, 0, 1], repeat=group_size):
            s = 0
            for c, w in zip(coeffs, group_weights):
                s += c * w
            group_sums.append((s, coeffs))
            
        groups.append(group_sums)
        
    # Combine G1 and G2 into M12
    M12 = []
    for s1, c1 in groups[0]:
        for s2, c2 in groups[1]:
            M12.append((s1 + s2, c1, c2))
            
    # Combine G3 and G4 into M34
    M34 = []
    for s3, c3 in groups[2]:
        for s4, c4 in groups[3]:
            M34.append((s3 + s4, c3, c4))
            
    # Sort M12 and M34 by sum
    M12.sort(key=lambda x: x[0])
    M34.sort(key=lambda x: x[0])
    
    # Sort G5 by sum
    G5 = groups[4]
    G5_sorted = sorted(G5, key=lambda x: x[0])
    
    best_sum = None
    best_coeffs = None
    
    i = 0
    j = len(M34) - 1
    len_M12 = len(M12)
    len_M34 = len(M34)
    
    for s5, c5 in G5_sorted:
        target = -s5
        
        # Move pointers to find closest sum to target
        # M12[i].sum + M34[j].sum should be close to target
        
        # Advance i while sum is too small
        while i < len_M12 - 1 and M12[i][0] + M34[j][0] < target:
            i += 1
            
        # Retreat j while sum is too large
        while j > 0 and M12[i][0] + M34[j][0] > target:
            j -= 1
            
        # Check candidates around current i, j
        # We check a small window to ensure we don't miss the min non-zero
        # if the exact closest is 0 or if pointers are stuck
        
        candidates = []
        for di in [-1, 0, 1]:
            ni = i + di
            if 0 <= ni < len_M12:
                for dj in [-1, 0, 1]:
                    nj = j + dj
                    if 0 <= nj < len_M34:
                        sum_12 = M12[ni][0]
                        sum_34 = M34[nj][0]
                        total_sum = sum_12 + sum_34 + s5
                        
                        if total_sum != 0:
                            candidates.append((abs(total_sum), total_sum, ni, nj))
        
        if candidates:
            # Find min abs sum
            min_abs, min_val, ni, nj = min(candidates, key=lambda x: x[0])
            
            if best_sum is None or abs(min_val) < abs(best_sum):
                best_sum = min_val
                # Reconstruct coeffs
                c1, c2 = M12[ni][1], M12[ni][2]
                c3, c4 = M34[nj][1], M34[nj][2]
                best_coeffs = c1 + c2 + c3 + c4 + c5
                
        if time.time() > deadline:
            break

    if best_coeffs is None:
        # Fallback
        fallback = [0] * 30
        fallback[1] = 1
        fallback[2] = -1
        return fallback
        
    return best_coeffs
