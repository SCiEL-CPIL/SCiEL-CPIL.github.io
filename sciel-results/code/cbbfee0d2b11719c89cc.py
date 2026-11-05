import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = 50.0  # 50s limit to be safe
    
    # 1. Calculate LCM of 1..30
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
        
    L = 1
    for i in range(1, 31):
        L = L * i // gcd(L, i)
        
    # 2. Define weights
    # Weights for A (indices 1..15)
    weights_A = np.array([L // i for i in range(1, 16)], dtype=np.int64)
    # Weights for B (indices 16..30)
    weights_B = np.array([L // i for i in range(16, 31)], dtype=np.int64)
    
    n_A = 15
    n_B = 15
    total_A = 3 ** n_A
    total_B = 3 ** n_B
    
    # Helper to generate sums and encoded vectors
    def generate_sums_and_vecs(weights, n):
        # Generate indices 0 to 3^n - 1
        # We need base-3 digits
        # Instead of creating a huge 2D array of digits, we can compute sums directly?
        # But we also need the encoded vector.
        
        # Create array of indices
        idxs = np.arange(total_A, dtype=np.int64)
        
        # Compute encoded vector directly from idxs?
        # The k-th digit of idx in base 3 is (idx // 3**k) % 3
        # We can compute the sum of weights * digits
        # Sum = sum_{k=0}^{n-1} weights[k] * ((idx // 3**k) % 3)
        
        # This is a loop over k (15 iterations), each doing vectorized ops on 1.4e7 elements.
        # This is efficient.
        
        sums = np.zeros(total_A, dtype=np.int64)
        enc_vecs = np.zeros(total_A, dtype=np.int64)
        
        pow3 = 1
        for k in range(n):
            digits = (idxs // pow3) % 3
            # Map 0,1,2 to -1,0,1 for sum calculation
            # sum += weights[k] * (digits - 1)
            sums += weights[k] * (digits - 1)
            
            # Encode: use digits 0,1,2 as is
            enc_vecs += digits * pow3
            
            pow3 *= 3
            
        return sums, enc_vecs

    # Check time
    if time.time() - start_time > deadline:
        return [0]*30

    sums_A, vecs_A = generate_sums_and_vecs(weights_A, n_A)
    sums_B, vecs_B = generate_sums_and_vecs(weights_B, n_B)
    
    if time.time() - start_time > deadline:
        return [0]*30

    # Sort A
    order_A = np.argsort(sums_A)
    sorted_sums_A = sums_A[order_A]
    sorted_vecs_A = vecs_A[order_A]
    
    # For each B, find best A
    # Target for A is -sums_B
    targets = -sums_B
    
    # Find insertion points
    # np.searchsorted returns index where target should be inserted to maintain order
    # So sorted_sums_A[idx-1] <= target < sorted_sums_A[idx]
    # Candidates are idx-1 and idx
    
    idxs = np.searchsorted(sorted_sums_A, targets)
    
    # Handle boundaries
    # If idx == 0, only right neighbor (idx) exists? No, idx is insertion point.
    # If target < sorted_sums_A[0], idx=0. Candidate is 0.
    # If target > sorted_sums_A[-1], idx=len. Candidate is len-1.
    
    left_idxs = np.clip(idxs - 1, 0, len(sorted_sums_A) - 1)
    right_idxs = np.clip(idxs, 0, len(sorted_sums_A) - 1)
    
    sums_left = sorted_sums_A[left_idxs] + sums_B
    sums_right = sorted_sums_A[right_idxs] + sums_B
    
    abs_left = np.abs(sums_left)
    abs_right = np.abs(sums_right)
    
    # We want non-zero minimum.
    # If a sum is 0, we treat it as invalid (infinity) for the purpose of finding the min non-zero.
    # But we need to be careful: if both are 0, we might miss a non-zero one nearby?
    # No, if sum_left == 0, it means sorted_sums_A[left_idx] == -sums_B.
    # The next best non-zero would be at left_idx - 1 or right_idx (if right_idx != left_idx).
    # Actually, if sum_left == 0, then abs_left = 0.
    # We set it to inf.
    # Then we pick the min of the remaining.
    
    abs_left[abs_left == 0] = np.iinfo(np.int64).max
    abs_right[abs_right == 0] = np.iinfo(np.int64).max
    
    # Determine which side is better for each B
    # We want the global minimum across all B
    # We can just compute the best abs sum for each B and find the global min
    
    best_abs_per_B = np.minimum(abs_left, abs_right)
    
    # Find the index in B that gives the global minimum
    global_min_idx = np.argmin(best_abs_per_B)
    
    min_val = best_abs_per_B[global_min_idx]
    
    if min_val == np.iinfo(np.int64).max:
        # This implies all sums were 0? Or something went wrong.
        # Fallback
        return [0]*28 + [1, -1]
        
    # Determine which side (left or right) was chosen for this global_min_idx
    # We need to reconstruct the vectors.
    
    if abs_left[global_min_idx] < abs_right[global_min_idx]:
        chosen_A_idx = left_idxs[global_min_idx]
        chosen_sum = sums_left[global_min_idx]
    else:
        chosen_A_idx = right_idxs[global_min_idx]
        chosen_sum = sums_right[global_min_idx]
        
    # Retrieve vectors
    enc_A = sorted_vecs_A[chosen_A_idx]
    enc_B = vecs_B[global_min_idx]
    
    # Decode enc_A to coefficients for A
    coeffs_A = []
    temp = enc_A
    for k in range(n_A):
        digit = temp % 3
        # Map 0,1,2 to -1,0,1
        coeff = digit - 1
        coeffs_A.append(coeff)
        temp //= 3
        
    # Decode enc_B to coefficients for B
    coeffs_B = []
    temp = enc_B
    for k in range(n_B):
        digit = temp % 3
        coeff = digit - 1
        coeffs_B.append(coeff)
        temp //= 3
        
    # Combine
    final_coeffs = coeffs_A + coeffs_B
    
    # Verify sum
    # S = sum(c_i / i)
    # K = sum(c_i * L/i)
    # We expect K == chosen_sum
    # Let's verify
    K_calc = 0
    for i in range(1, 31):
        c = final_coeffs[i-1]
        K_calc += c * (L // i)
        
    if K_calc != chosen_sum:
        # Mismatch? Should not happen.
        # Try to find a valid one?
        pass
        
    return final_coeffs

