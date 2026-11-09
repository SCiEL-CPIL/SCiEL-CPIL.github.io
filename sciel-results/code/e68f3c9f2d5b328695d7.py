import time
import numpy as np
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline to stay safely under 90s limit
    
    n = 30
    # LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = L * i // gcd(L, i)
    
    W = [L // (i+1) for i in range(n)]
    
    def time_left():
        return deadline - time.time()
    
    def generate_half_numpy(weights):
        m = len(weights)
        sums = np.array([0], dtype=np.int64)
        for w in weights:
            new_sums = np.empty(len(sums) * 3, dtype=np.int64)
            new_sums[0::3] = sums - w
            new_sums[1::3] = sums
            new_sums[2::3] = sums + w
            sums = new_sums
            if time_left() < 35:
                break
        return sums
    
    def index_to_coeffs(idx, m):
        """Convert flat index to coefficient tuple for m weights"""
        coeffs = [0] * m
        for pos in range(m - 1, -1, -1):
            choice = idx % 3
            idx //= 3
            coeffs[pos] = choice - 1  # 0->-1, 1->0, 2->1
        return coeffs
    
    def local_search(coeffs, W, max_rounds=50):
        """Improved local search with single, double, and triple moves"""
        current_sum = sum(c * w for c, w in zip(coeffs, W))
        n = len(W)
        
        for _ in range(max_rounds):
            if time_left() < 10:
                break
            improved = False
            
            # Single moves
            for pos in range(n):
                for delta in [-1, 1]:
                    new_val = coeffs[pos] + delta
                    if new_val in (-1, 0, 1):
                        new_sum = current_sum + delta * W[pos]
                        if new_sum != 0 and abs(new_sum) < abs(current_sum):
                            coeffs[pos] = new_val
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # Double moves (pairwise)
            for i in range(n):
                if time_left() < 10:
                    break
                for j in range(i+1, n):
                    for di in [-1, 1]:
                        if coeffs[i] + di not in (-1, 0, 1):
                            continue
                        for dj in [-1, 1]:
                            if coeffs[j] + dj not in (-1, 0, 1):
                                continue
                            new_sum = current_sum + di * W[i] + dj * W[j]
                            if new_sum != 0 and abs(new_sum) < abs(current_sum):
                                coeffs[i] += di
                                coeffs[j] += dj
                                current_sum = new_sum
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return coeffs, current_sum
    
    # Try multiple split points
    best_coeffs = None
    best_abs_sum = float('inf')
    
    # Primary split: 15/15
    splits_to_try = [15, 14, 16, 13, 17]
    
    for mid in splits_to_try:
        if time_left() < 25:
            break
        
        W1 = W[:mid]
        W2 = W[mid:]
        
        sums1 = generate_half_numpy(W1)
        if time_left() < 20:
            break
        
        sums2 = generate_half_numpy(W2)
        if time_left() < 10:
            break
        
        # Sort both arrays with original indices
        order1 = np.argsort(sums1)
        sums1_sorted = sums1[order1]
        order2 = np.argsort(sums2)
        sums2_sorted = sums2[order2]
        
        # Two-pointer scan to find min |s1 + s2| where s1 + s2 != 0
        n1 = len(sums1_sorted)
        n2 = len(sums2_sorted)
        i = 0
        j = n2 - 1
        best_val = float('inf')
        best_i = -1
        best_j = -1
        
        while i < n1 and j >= 0:
            s = sums1_sorted[i] + sums2_sorted[j]
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
            continue
        
        if best_val >= best_abs_sum:
            continue
        
        # Now backtrack to find coefficients
        orig_idx1 = order1[best_i]
        orig_idx2 = order2[best_j]
        
        coeffs1 = index_to_coeffs(int(orig_idx1), len(W1))
        coeffs2 = index_to_coeffs(int(orig_idx2), len(W2))
        
        coeffs = coeffs1 + coeffs2
        
        # Verify the sum
        total = sum(c * w for c, w in zip(coeffs, W))
        
        if total == 0:
            # Try nearby pairs
            found = False
            for di in range(-10, 11):
                for dj in range(-10, 11):
                    ni = best_i + di
                    nj = best_j + dj
                    if 0 <= ni < n1 and 0 <= nj < n2:
                        s = sums1_sorted[ni] + sums2_sorted[nj]
                        if s != 0:
                            oi1 = order1[ni]
                            oi2 = order2[nj]
                            c1 = index_to_coeffs(int(oi1), len(W1))
                            c2 = index_to_coeffs(int(oi2), len(W2))
                            c_all = c1 + c2
                            t = sum(c * w for c, w in zip(c_all, W))
                            if t != 0:
                                coeffs = c_all
                                total = t
                                found = True
                                break
                if found:
                    break
        
        if total != 0:
            abs_total = abs(total)
            if abs_total < best_abs_sum:
                best_abs_sum = abs_total
                best_coeffs = coeffs[:]
        
        # Local search on this candidate
        if time_left() > 15:
            refined_coeffs, refined_sum = local_search(coeffs, W, max_rounds=30)
            if refined_sum != 0:
                abs_refined = abs(refined_sum)
                if abs_refined < best_abs_sum:
                    best_abs_sum = abs_refined
                    best_coeffs = refined_coeffs
    
    # Final local search on the best candidate
    if best_coeffs is not None and time_left() > 5:
        best_coeffs, best_sum = local_search(best_coeffs, W, max_rounds=50)
    
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
