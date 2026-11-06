import numpy as np
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    # Calculate LCM of 1 to 30
    L = reduce(_lcm, range(1, 31))
    
    # Weights w_i = L / (i+1) for i=0..29
    weights = [L // (i + 1) for i in range(30)]
    
    # Split into two halves
    w1 = weights[:15]
    w2 = weights[15:]
    
    # Function to generate all sums and masks for a list of weights
    def gen_sums(ws):
        sums = np.array([0], dtype=np.int64)
        masks = np.array([0], dtype=np.int32)
        for i, w in enumerate(ws):
            # New sums: s + w, s - w, s
            # New masks: m + 1*3^i, m + 2*3^i, m
            # Choice 1 (+w) -> digit 1
            # Choice 2 (-w) -> digit 2
            # Choice 0 (0)  -> digit 0
            
            pow3 = 3 ** i
            new_sums = np.concatenate([sums + w, sums - w, sums])
            new_masks = np.concatenate([masks + pow3, masks + 2 * pow3, masks])
            sums = new_sums
            masks = new_masks
        return sums, masks

    sums1, masks1 = gen_sums(w1)
    sums2, masks2 = gen_sums(w2)
    
    # Sort sums1 and keep track of original indices
    sort_idx1 = np.argsort(sums1)
    sums1_sorted = sums1[sort_idx1]
    masks1_sorted = masks1[sort_idx1]
    
    # For each sum in sums2, find best match in sums1
    # We want s1 + s2 close to 0.
    # target = -s2
    targets = -sums2
    
    # Find insertion points
    indices = np.searchsorted(sums1_sorted, targets)
    
    # Candidate indices: indices, indices-1, indices+1
    # Handle boundaries
    cand_indices = np.concatenate([
        np.maximum(0, indices - 1),
        indices,
        np.minimum(len(sums1_sorted) - 1, indices + 1)
    ])
    
    # Get candidate sums and masks
    cand_sums = sums1_sorted[cand_indices]
    cand_masks = masks1_sorted[cand_indices]
    
    # Calculate total sums for these candidates
    # cand_sums is shape (3 * len(sums2),)
    # We need to map back to the original sums2
    # This is tricky with the concatenated array.
    # Let's iterate or use broadcasting carefully.
    
    # Better approach:
    # For each i in range(len(sums2)):
    #   idx = indices[i]
    #   Check idx-1, idx, idx+1
    # This loop is 14 million iterations. Might be slow in Python.
    # Let's try to vectorize the selection.
    
    # Create a flat array of candidate indices
    # Shape (3 * N)
    
    # We need to compute total_sum = cand_sums + sums2[original_index]
    # original_index for j-th element in cand_indices:
    # j=0..N-1: idx = j (from indices-1) -> orig = j
    # j=N..2N-1: idx = j-N (from indices) -> orig = j-N
    # j=2N..3N-1: idx = j-2N (from indices+1) -> orig = j-2N
    
    N = len(sums2)
    orig_indices = np.concatenate([
        np.arange(N),
        np.arange(N),
        np.arange(N)
    ])
    
    # Get corresponding sums2
    sums2_flat = sums2[orig_indices]
    
    total_sums = cand_sums + sums2_flat
    
    # Find minimum non-zero absolute value
    abs_sums = np.abs(total_sums)
    
    # Mask out zeros
    abs_sums_nonzero = abs_sums.copy()
    abs_sums_nonzero[total_sums == 0] = np.inf
    
    min_idx = np.argmin(abs_sums_nonzero)
    
    best_abs_val = abs_sums_nonzero[min_idx]
    
    # If all are zero (unlikely), we might need to handle that, but there are non-zero sums.
    
    # Recover the masks
    # min_idx is in the flattened candidate array
    cand_mask = cand_masks[min_idx]
    orig_idx = orig_indices[min_idx]
    sum2_mask = masks2[orig_idx]
    
    # Decode masks to coefficients
    def decode_mask(m, num_digits):
        coefs = []
        for i in range(num_digits):
            d = m % 3
            m //= 3
            if d == 0:
                coefs.append(0)
            elif d == 1:
                coefs.append(1)
            else:
                coefs.append(-1)
        return coefs
    
    coefs1 = decode_mask(cand_mask, 15)
    coefs2 = decode_mask(sum2_mask, 15)
    
    full_coefs = coefs1 + coefs2
    
    return full_coefs

if __name__ == "__main__":
    # Quick test
    res = construct()
    s = sum(c / (i + 1) for i, c in enumerate(res))
    print(f"Sum: {s}, Coefs: {res}")
