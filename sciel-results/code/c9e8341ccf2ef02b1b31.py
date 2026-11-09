"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    n = 30
    # Calculate LCM of 1..30
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
        
    # Weights for integer arithmetic: S = sum(c_i / i) = (1/LCM) * sum(c_i * LCM/i)
    # c_i in {-1, 0, 1}
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    # Use 4+4+4+4+4+4+4+4+2 split (9 groups: eight of size 4, one of size 2)
    # This reduces the meet-in-the-middle combination size significantly
    group_sizes = [4, 4, 4, 4, 4, 4, 4, 4, 2]
    groups = []
    idx = 0
    for size in group_sizes:
        groups.append(weights[idx:idx+size])
        idx += size
    
    # Generate all sums for each group
    def gen_sums(ws):
        sums = [0]
        for w in ws:
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums
    
    group_sums = []
    for g in groups:
        gs = gen_sums(g)
        gs.sort()
        group_sums.append(gs)
    
    if time.time() > deadline - 10:
        return [0]*28 + [1, -1]
    
    # Now we need to find the minimum non-zero sum across all combinations
    # We have 9 groups. We'll do a multi-level meet-in-the-middle.
    # Split into first 4 groups and last 5 groups.
    # First half: groups 0-3 (4 groups of 4) -> 3^4 = 81 elements per group, total 81^4 = ~43M
    # That's still too large. Let's do a different split.
    # Actually, let's just iterate through all combinations of the smaller groups.
    # With 8 groups of 4 and 1 group of 2:
    # 3^4 = 81 per group of 4
    # 3^2 = 9 per group of 2
    # Total combinations: 81^8 * 9 = 81^8 * 9
    # 81^8 = (3^4)^8 = 3^32 which is way too large.
    
    # Let me reconsider. The cell finding mentioned "4x7+2x1" split.
    # That means 4 groups of size 7 and 2 groups of size 1? No, 4*7 + 2*1 = 30.
    # So 4 groups of 7 elements and 2 groups of 1 element.
    # 3^7 = 2187 per group of 7
    # 3^1 = 3 per group of 1
    # Total: 2187^4 * 3^2 = 2187^4 * 9
    # 2187^4 = (3^7)^4 = 3^28 which is still too large.
    
    # Wait, let me re-read: "reduces the meet-in-the-middle combination size from ~500k to ~4.7M pairs"
    # So it's not enumerating all combinations, but doing a smarter meet-in-the-middle.
    
    # Let me try: split into two halves of 15 each, but use a more efficient algorithm.
    # Or use the two-pointer approach on sorted arrays as in the original.
    
    # Let me go back to the original two-half approach but optimize it.
    half = 15
    w1 = weights[:half]
    w2 = weights[half:]
    
    sums1 = gen_sums(w1)
    sums1.sort()
    
    if time.time() > deadline - 15:
        return [0]*28 + [1, -1]
    
    sums2 = gen_sums(w2)
    sums2.sort()
    
    # Two-pointer scan
    i = 0
    j = len(sums2) - 1
    n1 = len(sums1)
    n2 = len(sums2)
    
    best_abs = float('inf')
    best_a = None
    best_b = None
    
    while i < n1 and j >= 0:
        val = sums1[i] + sums2[j]
        if val == 0:
            if j > 0:
                v = sums1[i] + sums2[j-1]
                if v != 0:
                    av = abs(v)
                    if av < best_abs:
                        best_abs = av
                        best_a = sums1[i]
                        best_b = sums2[j-1]
                        if best_abs == 1:
                            break
            if i < n1 - 1:
                v = sums1[i+1] + sums2[j]
                if v != 0:
                    av = abs(v)
                    if av < best_abs:
                        best_abs = av
                        best_a = sums1[i+1]
                        best_b = sums2[j]
                        if best_abs == 1:
                            break
            i += 1
            j -= 1
        elif val > 0:
            if val < best_abs:
                best_abs = val
                best_a = sums1[i]
                best_b = sums2[j]
                if best_abs == 1:
                    break
            j -= 1
        else:
            av = -val
            if av < best_abs:
                best_abs = av
                best_a = sums1[i]
                best_b = sums2[j]
                if best_abs == 1:
                    break
            i += 1
    
    def find_coeffs(weights, target):
        n = len(weights)
        coeffs = [0] * n
        
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + weights[i]
            
        def dfs(idx, rem):
            if idx == n:
                return rem == 0
            if abs(rem) > suffix_sums[idx]:
                return False
            for c in [-1, 0, 1]:
                coeffs[idx] = c
                if dfs(idx + 1, rem - c * weights[idx]):
                    return True
            return False
            
        if dfs(0, target):
            return coeffs
        return None
        
    if best_a is not None and best_b is not None:
        c1 = find_coeffs(w1, best_a)
        c2 = find_coeffs(w2, best_b)
        if c1 and c2:
            return c1 + c2
            
    return [0] * (n - 2) + [1, -1]

# EVOLVE-BLOCK-END
