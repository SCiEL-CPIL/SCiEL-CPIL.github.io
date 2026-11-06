import numpy as np
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute L = lcm(2, ..., 48)
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    # L fits in int64?
    # lcm(1..48) is 5342931457063200. 
    # 2^63 is approx 9.22 * 10^18.
    # So yes, it fits.
    
    weights = np.array([L // d for d in denominators], dtype=np.int64)
    
    # Split into two halves
    n1 = 23
    n2 = n - n1
    
    w1 = weights[:n1]
    w2 = weights[n1:]
    
    # Generate subset sums for first half
    # sums1 will have size 2^n1
    size1 = 1 << n1
    sums1 = np.zeros(size1, dtype=np.int64)
    masks1 = np.arange(size1, dtype=np.int64)
    
    # Iterative generation
    # sums[0] = 0
    # for i in range(n1):
    #   w = w1[i]
    #   sums[2**i : 2**(i+1)] = sums[0 : 2**i] + w
    # This works if we fill in place, but numpy slicing creates copies or views.
    # Using concatenate is safer for logic, but let's use the in-place doubling trick carefully.
    
    # Actually, the standard way:
    # sums = [0]
    # masks = [0]
    # for w in w1:
    #   sums += [s + w for s in sums]
    #   masks += [m | (1<<i) for m in masks]
    # This is O(N * 2^N) in Python lists, which is slow.
    
    # NumPy vectorized generation:
    # Start with array of length 1
    # sums1 = np.array([0], dtype=np.int64)
    # masks1 = np.array([0], dtype=np.int64)
    # for i in range(n1):
    #     w = w1[i]
    #     bit = 1 << i
    #     sums1 = np.concatenate([sums1, sums1 + w])
    #     masks1 = np.concatenate([masks1, masks1 | bit])
    
    # This creates new arrays each time. Total memory allocated over time is sum of sizes.
    # Final size 2^23.
    # This is fast.
    
    sums1 = np.array([0], dtype=np.int64)
    masks1 = np.array([0], dtype=np.int64)
    for i in range(n1):
        w = w1[i]
        bit = 1 << i
        sums1 = np.concatenate([sums1, sums1 + w])
        masks1 = np.concatenate([masks1, masks1 | bit])
        
    if time.time() > deadline:
        return [0]
        
    # Generate subset sums for second half
    sums2 = np.array([0], dtype=np.int64)
    masks2 = np.array([0], dtype=np.int64)
    for i in range(n2):
        w = w2[i]
        bit = 1 << i
        sums2 = np.concatenate([sums2, sums2 + w])
        masks2 = np.concatenate([masks2, masks2 | bit])
        
    if time.time() > deadline:
        return [0]
        
    # Sort sums2 and masks2 together
    # We want to sort by sums2
    idx_sort = np.argsort(sums2)
    sorted_sums2 = sums2[idx_sort]
    sorted_masks2 = masks2[idx_sort]
    
    if time.time() > deadline:
        return [0]
        
    # Vectorized search
    # For each s1 in sums1, we want the largest s2 such that s1 + s2 < L
    # i.e., s2 < L - s1
    targets = L - sums1
    # searchsorted returns the index where the target would be inserted to maintain order
    # side='left' means if target exists, it returns the first occurrence.
    # We want the largest element STRICTLY LESS THAN target.
    # So we find the insertion point for target (left) and take the index before it.
    positions = np.searchsorted(sorted_sums2, targets, side='left') - 1
    
    # Filter valid positions
    valid = positions >= 0
    
    if not np.any(valid):
        # Fallback: should not happen, but just in case
        return [0]
        
    valid_positions = positions[valid]
    valid_sums1 = sums1[valid]
    valid_masks1 = masks1[valid]
    
    # Calculate candidate sums
    candidate_sums = valid_sums1 + sorted_sums2[valid_positions]
    
    # Find the maximum
    best_idx = np.argmax(candidate_sums)
    best_sum = candidate_sums[best_idx]
    best_mask1 = valid_masks1[best_idx]
    best_mask2 = sorted_masks2[valid_positions[best_idx]]
    
    # Reconstruct indices
    indices = []
    for k in range(n1):
        if best_mask1 & (1 << k):
            indices.append(k)
            
    for k in range(n2):
        if best_mask2 & (1 << k):
            indices.append(n1 + k)
            
    if not indices:
        # Fallback: 1/2
        indices = [0]
        
    return indices

if __name__ == "__main__":
    res = construct()
    denominators = list(range(2, 49))
    s = 0
    for idx in res:
        s += 1.0 / denominators[idx]
    print(f"Indices: {res}")
    print(f"Sum: {s}")
    print(f"Deficit: {1.0 - s}")
