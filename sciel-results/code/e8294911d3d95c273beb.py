import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline to stay under 90s limit

    n = 30
    # LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = L * i // gcd(L, i)
    
    W = [L // (i+1) for i in range(n)]
    mid = n // 2  # 15
    W1 = W[:mid]
    W2 = W[mid:]
    
    # Generate all 3^15 sums for first half using numpy
    def generate_half_sums_numpy(weights):
        m = len(weights)
        total = 3 ** m
        # We'll build the sums array iteratively
        # Start with [0]
        sums = np.array([0], dtype=np.int64)
        for w in weights:
            # For each existing sum, create three new sums: s-w, s, s+w
            new_sums = np.empty(len(sums) * 3, dtype=np.int64)
            new_sums[0::3] = sums - w
            new_sums[1::3] = sums
            new_sums[2::3] = sums + w
            sums = new_sums
            if time.time() > deadline - 30:
                break
        return sums
    
    # Generate sums for both halves
    sums1 = generate_half_sums_numpy(W1)
    if time.time() > deadline - 20:
        return [0] * 28 + [1, -1]
    
    sums2 = generate_half_sums_numpy(W2)
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]
    
    # Sort both arrays
    sums1.sort()
    sums2.sort()
    
    # Two-pointer scan to find min |s1 + s2| where s1 + s2 != 0
    n1 = len(sums1)
    n2 = len(sums2)
    i = 0
    j = n2 - 1
    best_val = float('inf')
    best_i = -1
    best_j = -1
    
    while i < n1 and j >= 0:
        s = sums1[i] + sums2[j]
        if s != 0:
            abs_s = abs(s)
            if abs_s < best_val:
                best_val = abs_s
                best_i = i
                best_j = j
                if best_val == 1:
                    break
        if s > 0:
            j -= 1
        elif s < 0:
            i += 1
        else:
            j -= 1
    
    if best_val == float('inf'):
        return [0] * 28 + [1, -1]
    
    # Now we need to find which coefficient combination produces sums1[best_i] and sums2[best_j]
    # We need to backtrack through the generation to find the coefficients.
    # This is tricky with numpy. Let's use a different approach:
    # Store the indices during generation, or re-search.
    
    # Alternative: Since we know the target sum for each half, we can search for it.
    # But finding which of 3^15 combinations gives a specific sum is hard without storing indices.
    
    # Better approach: Store indices as well. But that doubles memory.
    # Or: Use a recursive search to find coefficients that produce the target sum.
    
    # Let's try a different strategy: store both sums and indices.
    # Actually, let's just do a DFS to find coefficients for a given target sum.
    # This might be slow if the target sum is rare, but let's try.
    
    def find_coeffs_for_sum(weights, target, max_depth=15):
        """Find coefficients c[i] in {-1,0,1} such that sum(c[i]*weights[i]) == target"""
        coeffs = [0] * len(weights)
        
        def dfs(pos, current_sum):
            if time.time() > deadline - 5:
                return False
            if pos == len(weights):
                return current_sum == target
            w = weights[pos]
            # Try c = -1
            coeffs[pos] = -1
            if dfs(pos + 1, current_sum - w):
                return True
            # Try c = 0
            coeffs[pos] = 0
            if dfs(pos + 1, current_sum):
                return True
            # Try c = 1
            coeffs[pos] = 1
            if dfs(pos + 1, current_sum + w):
                return True
            return False
        
        if dfs(0, 0):
            return coeffs
        return None
    
    target1 = sums1[best_i]
    target2 = sums2[best_j]
    
    coeffs1 = find_coeffs_for_sum(W1, target1)
    if time.time() > deadline - 3:
        return [0] * 28 + [1, -1]
    
    coeffs2 = find_coeffs_for_sum(W2, target2)
    if time.time() > deadline - 1:
        return [0] * 28 + [1, -1]
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
    
    coeffs = coeffs1 + coeffs2
    return coeffs
