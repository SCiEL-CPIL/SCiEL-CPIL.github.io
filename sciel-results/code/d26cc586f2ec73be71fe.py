import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
        
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    half = 15
    left_weights = weights[:half]
    right_weights = weights[half:]
    
    # Generate left sums and paths (encoded in base 3)
    left_sums = np.array([0], dtype=np.int64)
    left_paths = np.array([0], dtype=np.int32)
    
    for w in left_weights:
        # Expand sums: 0, +w, -w
        # paths: 0, 1, 2 (where 0 is coeff 0, 1 is coeff 1, 2 is coeff -1)
        # New path = old_path * 3 + digit
        new_sums = np.concatenate([left_sums, left_sums + w, left_sums - w])
        new_paths = np.concatenate([left_paths * 3 + 0, left_paths * 3 + 1, left_paths * 3 + 2])
        left_sums = new_sums
        left_paths = new_paths
        
    if time.time() > deadline:
        return [0]*28 + [1, -1]

    # Generate right sums and paths
    right_sums = np.array([0], dtype=np.int64)
    right_paths = np.array([0], dtype=np.int32)
    
    for w in right_weights:
        new_sums = np.concatenate([right_sums, right_sums + w, right_sums - w])
        new_paths = np.concatenate([right_paths * 3 + 0, right_paths * 3 + 1, right_paths * 3 + 2])
        right_sums = new_sums
        right_paths = new_paths

    if time.time() > deadline:
        return [0]*28 + [1, -1]

    # Sort left sums and keep track of original indices
    left_order = np.argsort(left_sums)
    left_sorted = left_sums[left_order]
    left_paths_sorted = left_paths[left_order]
    
    # Find best pair
    # targets = -right_sums
    targets = -right_sums
    
    # Find insertion points in sorted left array
    insert_pos = np.searchsorted(left_sorted, targets)
    
    # Check positions: insert_pos-1, insert_pos, insert_pos+1
    n_right = len(right_sums)
    candidates = np.empty((3, n_right), dtype=np.int64)
    candidates[0] = np.clip(insert_pos - 1, 0, len(left_sorted) - 1)
    candidates[1] = np.clip(insert_pos, 0, len(left_sorted) - 1)
    candidates[2] = np.clip(insert_pos + 1, 0, len(left_sorted) - 1)
    
    # Get the actual left values for each candidate
    candidate_left_vals = left_sorted[candidates]
    
    # Compute the sums: candidate_left_val + right_val
    sums = candidate_left_vals + right_sums[np.newaxis, :]
    
    # Mask out zeros
    nonzero_mask = sums != 0
    
    # For positions where the sum is zero, set absolute value to infinity
    abs_sums = np.where(nonzero_mask, np.abs(sums), np.int64(2**62))
    
    # Find the minimum across all 3 candidates for each right
    min_abs_per_right = np.min(abs_sums, axis=0)
    
    # Find the overall minimum
    global_min_idx = np.argmin(min_abs_per_right)
    best_right_idx = global_min_idx
    
    # Find which candidate (0, 1, or 2) gave the minimum for this right
    candidate_idx = np.argmin(abs_sums[:, global_min_idx])
    best_left_sorted_idx = candidates[candidate_idx, global_min_idx]
    
    # Map back to original left index
    best_left_idx = left_order[best_left_sorted_idx]
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # Decode paths
    left_path_int = left_paths[best_left_idx]
    right_path_int = right_paths[best_right_idx]
    
    left_coeffs = []
    for _ in range(half):
        left_coeffs.append(left_path_int % 3 - 1) # 0->-1? No.
        # Encoding: 0->0, 1->1, 2->-1
        # digit = path % 3
        # if digit 0: 0
        # if digit 1: 1
        # if digit 2: -1
        # So: digit - 1 gives -1, 0, 1?
        # 0->-1, 1->0, 2->1.
        # We want 0->0, 1->1, 2->-1.
        # Let's stick to explicit mapping.
        d = left_path_int % 3
        if d == 0:
            left_coeffs.append(0)
        elif d == 1:
            left_coeffs.append(1)
        else:
            left_coeffs.append(-1)
        left_path_int //= 3
        
    right_coeffs = []
    for _ in range(n - half):
        d = right_path_int % 3
        if d == 0:
            right_coeffs.append(0)
        elif d == 1:
            right_coeffs.append(1)
        else:
            right_coeffs.append(-1)
        right_path_int //= 3
        
    # Reverse because we built paths by appending to the end (LSB is last added?)
    # Wait.
    # Start: path=0.
    # Step 1: path = 0*3 + d1 = d1.
    # Step 2: path = d1*3 + d2 = 3*d1 + d2.
    # So d1 is the most significant digit?
    # No, d1 is the coefficient for the FIRST weight.
    # When decoding, we take % 3 first, which gives d2 (last added).
    # So the decoded list is in reverse order of weights.
    left_coeffs.reverse()
    right_coeffs.reverse()
    
    if len(left_coeffs) != half or len(right_coeffs) != (n - half):
        return [0]*28 + [1, -1]
    
    return left_coeffs + right_coeffs
