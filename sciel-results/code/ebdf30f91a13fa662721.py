import numpy as np
from math import gcd
from functools import reduce
import time

def construct():
       start_time = time.time()
       deadline = 80.0  # Leave 10s margin

       # Calculate LCM of 1..30
       def lcm(a, b):
           return a * b // gcd(a, b)
       
       L = reduce(lcm, range(1, 31))
       weights = [L // i for i in range(1, 31)]
       
       # Split into two halves
       w1 = np.array(weights[:15], dtype=np.int64)
       w2 = np.array(weights[15:], dtype=np.int64)
       
       n1 = 15
       n2 = 15
       size1 = 3 ** n1
       size2 = 3 ** n2
       
       # Generate sums for first half
       # indices1 = 0 .. 3^15 - 1
       indices1 = np.arange(size1, dtype=np.int64)
       sums1 = np.zeros(size1, dtype=np.int64)
       
       for k in range(n1):
           power = 3 ** k
           digit = (indices1 // power) % 3
           # Map digit 0->-1, 1->0, 2->1
           coeff = digit - 1
           sums1 += coeff * w1[k]
           
       if time.time() > deadline:
           # Fallback
           return [0]*28 + [1, -1]
       
       # Sort sums1
       order = np.argsort(sums1)
       sorted_sums1 = sums1[order]
       sorted_indices1 = indices1[order]
       
       if time.time() > deadline:
           return [0]*28 + [1, -1]
       
       # Generate sums for second half
       indices2 = np.arange(size2, dtype=np.int64)
       sums2 = np.zeros(size2, dtype=np.int64)
       
       for k in range(n2):
           power = 3 ** k
           digit = (indices2 // power) % 3
           coeff = digit - 1
           sums2 += coeff * w2[k]
           
       if time.time() > deadline:
           return [0]*28 + [1, -1]
       
       # Find best match
       targets = -sums2
       pos = np.searchsorted(sorted_sums1, targets)
       
       # Check pos-1 and pos
       pos_left = np.clip(pos - 1, 0, len(sorted_sums1) - 1)
       pos_right = np.clip(pos, 0, len(sorted_sums1) - 1)
       
       cand_left = sorted_sums1[pos_left]
       cand_right = sorted_sums1[pos_right]
       
       diff_left = np.abs(cand_left + sums2)
       diff_right = np.abs(cand_right + sums2)
       
       # Stack and find min
       all_diffs = np.stack([diff_left, diff_right], axis=1)
       best_local_idx = np.argmin(all_diffs, axis=1)
       best_diff = np.min(all_diffs, axis=1)
       
       best_pos = np.where(best_local_idx == 0, pos_left, pos_right)
       
       # Exclude zero sums
       non_zero_mask = best_diff > 0
       
       if np.any(non_zero_mask):
           # Find index with minimum non-zero difference
           min_diff_val = np.min(best_diff[non_zero_mask])
           # Find the first index that achieves this minimum
           candidates = np.where(non_zero_mask & (best_diff == min_diff_val))[0]
           best_s2_idx = candidates[0]
           best_s1_pos = best_pos[best_s2_idx]
       else:
           # Fallback: find any non-zero sum by checking more neighbors or brute force on small set
           # This case is extremely unlikely
           best_s2_idx = 0
           best_s1_pos = 0
           # Just pick something small
           # For safety, let's just return a known small sum
           return [0]*28 + [1, -1]
           
       # Reconstruct coefficients
       s1_idx = sorted_indices1[best_s1_pos]
       s2_idx = best_s2_idx
       
       coeffs1 = []
       temp = s1_idx
       for k in range(n1):
           digit = temp % 3
           temp //= 3
           coeffs1.append(digit - 1)
           
       coeffs2 = []
       temp = s2_idx
       for k in range(n2):
           digit = temp % 3
           temp //= 3
           coeffs2.append(digit - 1)
           
       coeffs = coeffs1 + coeffs2
       
       # Verify
       s = sum(c * w for c, w in zip(coeffs, weights))
       if s == 0:
           # Should not happen
           return [0]*28 + [1, -1]
           
       return coeffs
