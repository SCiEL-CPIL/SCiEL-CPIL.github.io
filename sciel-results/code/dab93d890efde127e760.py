import time
import math
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 90.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = reduce(lcm, denominators, 1)
    
    # Weights W_i = L // d_i
    W = [L // d for d in denominators]
    
    # Split into two halves
    # A: first 23, B: last 24
    n_a = 23
    n_b = n - n_a
    
    indices_a = list(range(n_a))
    indices_b = list(range(n_a, n))
    
    # Generate sums for B
    # We want to find max S_B such that S_A + S_B < L
    # So we sort S_B and binary search for L - S_A
    
    sums_b = []
    masks_b = []
    
    # Generate all 2^24 subsets of B
    # Using iterative bit manipulation or recursion?
    # Iterative is faster.
    # We can use a list of size 2^n_b
    
    size_b = 1 << n_b
    # Pre-allocate lists?
    # sums_b = [0] * size_b
    # masks_b = [0] * size_b
    
    # Efficient generation
    # s[0] = 0
    # s[mask] = s[mask without lowest set bit] + W[index of lowest set bit]
    
    # Let's just use a loop
    # To save memory, we might only store sums and then re-derive masks?
    # No, re-deriving masks for 16M items is slow.
    # But storing 16M masks is memory heavy.
    
    # Compromise: Store (sum, mask) in a single list of tuples?
    # Or two parallel lists.
    
    # Let's try parallel lists.
    sums_b = [0] * size_b
    masks_b = [0] * size_b
    
    # sums_b[0] = 0, masks_b[0] = 0
    for mask in range(1, size_b):
        # Find lowest set bit
        # lsb = mask & -mask
        # idx = lsb.bit_length() - 1
        # prev_mask = mask ^ lsb
        # sums_b[mask] = sums_b[prev_mask] + W[n_a + idx]
        # masks_b[mask] = mask (we just need the mask itself, no need to store it separately if we iterate by mask)
        
        # Actually, we don't need to store masks_b if we iterate by mask later?
        # But we need to sort by sum.
        # If we sort sums_b, we lose the association with mask.
        # So we must sort (sum, mask) pairs.
        
        pass

    # Better approach: Create list of (sum, mask)
    # This uses more memory but is simpler.
    # 16M tuples.
    
    pairs_b = []
    # sums_b[0] = 0
    # We can generate sums in an array first, then zip with range?
    
    # Let's optimize the generation of sums for B
    # s_b = [0] * size_b
    s_b = [0] * size_b
    for mask in range(1, size_b):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        s_b[mask] = s_b[prev] + W[n_a + idx]
        
    # Now create list of (sum, mask)
    # To save memory, we can store just the sums and sort indices?
    # No, we need to retrieve the mask.
    
    # Let's create the list of tuples
    # This might be slow.
    
    # Alternative:
    # Sort the range of masks based on s_b[mask]?
    # sorted_masks = sorted(range(size_b), key=lambda m: s_b[m])
    # This creates a list of 16M integers (masks) and sorts them.
    # Then we can binary search on s_b[sorted_masks[mid]].
    # This saves the tuple overhead!
    # Memory: 16M ints (s_b) + 16M ints (sorted_masks) = 16M * 28 * 2 = 900MB.
    # Plus the list overhead.
    
    sorted_masks = list(range(size_b))
    sorted_masks.sort(key=lambda m: s_b[m])
    
    # Now iterate over A
    size_a = 1 << n_a
    s_a = [0] * size_a
    for mask in range(1, size_a):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        s_a[mask] = s_a[prev] + W[idx]
        
    best_deficit_num = L # Deficit = L - (S_A + S_B). Minimize this.
    best_mask_a = 0
    best_mask_b = 0
    found = False
    
    # Binary search on sorted_masks
    # We want max s_b[m] such that s_a[ma] + s_b[m] < L
    # i.e., s_b[m] < L - s_a[ma]
    
    import bisect
    
    # Extract sorted sums for faster access?
    # sorted_sums = [s_b[m] for m in sorted_masks]
    # This adds another 16M list.
    # Let's just use s_b[sorted_masks[mid]] in the bisect key?
    # bisect doesn't support custom keys directly in older Python, but we can use a helper.
    # Or we can just extract sorted_sums.
    # Memory check: 3 lists of 16M ints. ~1.4 GB.
    # Risky but likely ok.
    
    sorted_sums = [s_b[m] for m in sorted_masks]
    
    for ma in range(size_a):
        # Check time
        if time.time() > deadline - 5.0:
            break
            
        target = L - s_a[ma]
        # Find rightmost index in sorted_sums where value < target
        # bisect_left returns index of first element >= target
        # So we want bisect_left - 1
        idx = bisect.bisect_left(sorted_sums, target)
        
        if idx > 0:
            mb = sorted_masks[idx - 1]
            sum_total = s_a[ma] + s_b[mb]
            # sum_total < L is guaranteed by bisect
            deficit = L - sum_total
            if deficit < best_deficit_num:
                best_deficit_num = deficit
                best_mask_a = ma
                best_mask_b = mb
                found = True
                
    if not found:
        # Fallback: should not happen as empty set is not allowed?
        # "Return nonempty distinct... indices"
        # If no subset found, return a valid single element?
        # 1/2 < 1. Deficit 1/2.
        # But we found one.
        return [0]
        
    # Reconstruct indices
    result = []
    for i in range(n_a):
        if best_mask_a & (1 << i):
            result.append(i)
            
    for i in range(n_b):
        if best_mask_b & (1 << i):
            result.append(n_a + i)
            
    return result
