"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # 2. Calculate weights W_i = L // i
    weights = [L // i for i in range(1, 31)]
    
    # 3. Split weights: try 14+16 split for better balance
    n1 = 14
    n2 = 16
    half1_weights = weights[:n1]
    half2_weights = weights[n1:]
    
    def generate_sums_with_indices(ws):
        """
        Generate all possible signed sums for the given weights.
        Returns (sums_array, indices_array) where indices_array[i] is the base-3 index
        that produced sums_array[i].
        """
        sums = np.array([0], dtype=np.int64)
        indices = np.array([0], dtype=np.int64)
        for w in ws:
            n = len(sums)
            new_sums = np.empty(3 * n, dtype=np.int64)
            new_indices = np.empty(3 * n, dtype=np.int64)
            
            new_sums[0:n] = sums - w
            new_indices[0:n] = indices * 3 + 0
            
            new_sums[n:2*n] = sums
            new_indices[n:2*n] = indices * 3 + 1
            
            new_sums[2*n:3*n] = sums + w
            new_indices[2*n:3*n] = indices * 3 + 2
            
            sums = new_sums
            indices = new_indices
        return sums, indices
    
    # Generate sums and indices for both halves
    sums1, idx1 = generate_sums_with_indices(half1_weights)
    sums2, idx2 = generate_sums_with_indices(half2_weights)
    
    if time.time() > deadline - 5:
        return [0] * 28 + [1, -1]
    
    # 4. Sort sums2 and keep track of original indices
    sort_order = np.argsort(sums2)
    sorted_sums2 = sums2[sort_order]
    sorted_idx2 = idx2[sort_order]
    
    # 5. For each sum in half1, find the best matching sum in half2
    neg_sums1 = -sums1
    insert_pos = np.searchsorted(sorted_sums2, neg_sums1, side='left')
    
    n_sums2 = len(sorted_sums2)
    
    # Check candidates at insert_pos and insert_pos-1
    pos1 = np.clip(insert_pos, 0, n_sums2 - 1)
    pos2 = np.clip(insert_pos - 1, 0, n_sums2 - 1)
    
    cand1 = sorted_sums2[pos1]
    cand2 = sorted_sums2[pos2]
    
    total1 = sums1 + cand1
    total2 = sums1 + cand2
    
    abs_total1 = np.where(total1 != 0, np.abs(total1), np.iinfo(np.int64).max)
    abs_total2 = np.where(total2 != 0, np.abs(total2), np.iinfo(np.int64).max)
    
    # Combine: for each i, take the better of the two candidates
    best_abs = np.minimum(abs_total1, abs_total2)
    best_pos = np.where(abs_total1 <= abs_total2, pos1, pos2)
    
    # Find the global minimum
    min_abs = np.min(best_abs)
    
    if min_abs == np.iinfo(np.int64).max:
        return [0] * 28 + [1, -1]
    
    k1_idx = np.argmin(best_abs)
    pos_in_sorted = best_pos[k1_idx]
    orig_pos = sort_order[pos_in_sorted]
    idx_in_sums2 = orig_pos
    idx_in_sums1 = k1_idx
    
    # 6. Decode indices to coefficients
    def decode_index(k, n_weights):
        coeffs = []
        for _ in range(n_weights):
            digit = k % 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
            k //= 3
        coeffs.reverse()
        return coeffs

    coeffs1 = decode_index(int(idx1[idx_in_sums1]), n1)
    coeffs2 = decode_index(int(idx2[idx_in_sums2]), n2)
    
    result = coeffs1 + coeffs2
    
    # 7. Local search polish: try flipping individual coefficients
    def compute_sum(coeffs):
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * (L // (i + 1))
        return s
    
    current_sum = compute_sum(result)
    if current_sum == 0:
        return [0] * 28 + [1, -1]
    
    current_abs = abs(current_sum)
    
    # Try flipping each coefficient to see if we get closer to zero
    for _ in range(3):  # A few rounds of local search
        improved = False
        for i in range(30):
            old_c = result[i]
            for new_c in [-1, 0, 1]:
                if new_c == old_c:
                    continue
                result[i] = new_c
                new_sum = compute_sum(result)
                if new_sum != 0 and abs(new_sum) < current_abs:
                    current_abs = abs(new_sum)
                    current_sum = new_sum
                    improved = True
                    break
                else:
                    result[i] = old_c
        if not improved or time.time() > deadline - 2:
            break
    
    # Final verification
    final_sum = compute_sum(result)
    if final_sum == 0:
        return [0] * 28 + [1, -1]
    
    return result

# EVOLVE-BLOCK-END
