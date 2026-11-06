import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    denoms = list(range(2, 49))
    D = 1
    for d in denoms:
        D = D * d // math.gcd(D, d)

    w = [D // d for d in denoms]
    n = len(w)
    n1 = n // 2
    n2 = n - n1

    w1 = w[:n1]
    w2 = w[n1:]

    MASK_BITS = n2
    MASK_LIMIT = (1 << MASK_BITS) - 1

    # Generate L2 (smaller half, n2=24? No, n1=23, n2=24. Let's swap to make L2 smaller for sorting?
    # If n1=24, n2=23.
    # Let's stick to n1 = 23, n2 = 24.
    # L2 size 2^24. L1 size 2^23.
    # Sort L2. Iterate L1 with bisect?
    # Or Sort both and two pointer?
    # If we filter, sizes are smaller.
    # Let's generate L2 first.
    
    def generate_packed(weights, mask_bits):
        limit_mask = (1 << mask_bits) - 1
        L = [0]
        for i in range(len(weights)):
            wi = weights[i]
            mi = 1 << i
            delta = (wi << mask_bits) | mi
            # Filter: keep x if sum < D
            # New items: x + delta if sum < D
            # We need D here.
            new_items = []
            for x in L:
                val = x + delta
                # Check sum part
                if (val >> mask_bits) < D:
                    new_items.append(val)
            L.extend(new_items)
            if time.time() > deadline:
                return L
        return L

    # Wait, the mask bits for L1 and L2 are different if we pack them differently?
    # In the previous plan, I used the same MASK_BITS for both?
    # If I use different mask bits, I can't just combine masks easily?
    # Yes I can. m1 is lower bits, m2 is higher bits.
    # But the packing shift must be consistent with the mask width.
    # If L1 uses mask_bits_1 and L2 uses mask_bits_2.
    # Packed1 = (sum1 << mask_bits_1) | m1
    # Packed2 = (sum2 << mask_bits_2) | m2
    # To compare sums, we just shift by their respective mask bits.
    # To combine, m1 is in bits 0..mask_bits_1-1.
    # m2 is in bits 0..mask_bits_2-1.
    # Final mask: m1 | (m2 << mask_bits_1).
    
    # However, for the two-pointer approach, we need to sort by sum.
    # If we pack with different shifts, sorting by integer value sorts by (sum, mask) for that specific packing.
    # Since sum is the higher bits, sorting by integer value sorts by sum primarily.
    # This works.
    
    # Let's use n1=23, n2=24.
    # MASK_BITS_1 = 23
    # MASK_BITS_2 = 24
    
    MASK_BITS_1 = n1
    MASK_BITS_2 = n2
    MASK_LIMIT_1 = (1 << MASK_BITS_1) - 1
    MASK_LIMIT_2 = (1 << MASK_BITS_2) - 1

    def generate_packed(weights, mask_bits, mask_limit):
        L = [0]
        for i in range(len(weights)):
            wi = weights[i]
            mi = 1 << i
            delta = (wi << mask_bits) | mi
            new_items = []
            # Optimization: local vars
            D_local = D
            mb = mask_bits
            for x in L:
                val = x + delta
                if (val >> mb) < D_local:
                    new_items.append(val)
            L.extend(new_items)
            if time.time() > deadline:
                break
        return L

    L2 = generate_packed(w2, MASK_BITS_2, MASK_LIMIT_2)
    if time.time() > deadline:
        return [0]
        
    L1 = generate_packed(w1, MASK_BITS_1, MASK_LIMIT_1)
    if time.time() > deadline:
        return [0]

    L1.sort()
    L2.sort()
    
    if time.time() > deadline:
        return [0]

    best_sum = -1
    best_p1 = 0
    best_p2 = 0
    
    i = 0
    j = len(L2) - 1
    len1 = len(L1)
    len2 = len(L2)
    
    mb1 = MASK_BITS_1
    mb2 = MASK_BITS_2
    D_local = D
    
    while i < len1 and j >= 0:
        if time.time() > deadline:
            break
            
        p1 = L1[i]
        s1 = p1 >> mb1
        
        p2 = L2[j]
        s2 = p2 >> mb2
        
        total_s = s1 + s2
        
        if total_s < D_local:
            if total_s > best_sum:
                best_sum = total_s
                best_p1 = p1
                best_p2 = p2
            i += 1
        else:
            j -= 1
            
    if best_sum < 0:
        return [0]
        
    m1 = best_p1 & MASK_LIMIT_1
    m2 = best_p2 & MASK_LIMIT_2
    
    result = []
    for i in range(n1):
        if (m1 >> i) & 1:
            result.append(i)
    for i in range(n2):
        if (m2 >> i) & 1:
            result.append(n1 + i)
            
    return result
