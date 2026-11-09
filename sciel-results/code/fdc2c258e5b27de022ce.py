import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s wall-clock limit

    n = 210
    max_val = 10210

    # Target: product(selected) * 210 must be perfect square
    # 210 = 2 * 3 * 5 * 7 (square-free)
    # Using bitmask signatures: each prime gets a bit position
    # We need XOR of all selected signatures XOR target_mask = 0

    # Assign bit positions to primes
    # Primes up to ~10210: we need to handle all primes
    # Use a dictionary mapping prime -> bit index
    # For efficiency, only primes that appear in numbers 211..max_val matter
    
    # Build SPF
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Collect all primes that appear
    primes_set = set()
    for x in range(211, max_val + 1):
        temp = x
        while temp > 1:
            p = spf[temp]
            primes_set.add(p)
            while temp % p == 0:
                temp //= p
    
    # Assign bit positions
    primes_list = sorted(primes_set)
    prime_to_bit = {p: i for i, p in enumerate(primes_list)}
    
    # Target mask: bits for primes 2, 3, 5, 7
    target_mask = 0
    for p in [2, 3, 5, 7]:
        if p in prime_to_bit:
            target_mask |= (1 << prime_to_bit[p])
        else:
            # Should not happen since 2,3,5,7 are small
            target_mask |= (1 << 0)  # fallback

    def get_sig_mask(x):
        mask = 0
        temp = x
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                if p in prime_to_bit:
                    mask |= (1 << prime_to_bit[p])
        return mask

    # Precompute signature masks for all numbers in range
    sig_masks = {}
    for x in range(211, max_val + 1):
        sig_masks[x] = get_sig_mask(x)

    best_value = float('inf')
    best_answer = None

    # Check 1-element solutions
    for k in range(2, 100):
        x = 210 * k * k
        if x > max_val:
            break
        val = x - 210
        if val < best_value:
            best_value = val
            best_answer = [x]

    # Check 2-element solutions: sig(a) XOR sig(b) XOR target_mask == 0
    # i.e., sig(b) = sig(a) XOR target_mask
    # Iterate U from 212 upward, maintain map from mask to smallest number with that mask
    mask_to_num = {}
    for U in range(212, max_val + 1):
        if time.time() > deadline - 10:
            break
        s = sig_masks[U]
        complement = s ^ target_mask
        if complement in mask_to_num:
            a = mask_to_num[complement]
            b = U
            val = U - 210
            if val < best_value:
                best_value = val
                best_answer = [a, b]
            break
        if s not in mask_to_num:
            mask_to_num[s] = U

    # Check 3-element solutions if 2-element didn't find something with value < some threshold
    # For 3 elements: sig(a) XOR sig(b) XOR sig(c) XOR target_mask == 0
    # i.e., sig(c) = sig(a) XOR sig(b) XOR target_mask
    # Strategy: iterate c from 213 upward, for each c check if there exist a < b < c
    # with sig(a) XOR sig(b) = sig(c) XOR target_mask
    # Maintain a map from mask to smallest number seen so far
    # When adding c, for each previously seen mask m1, check if m1 XOR sig(c) XOR target_mask is also seen
    
    # More efficient: maintain map mask -> smallest number
    # For each new c, for each existing mask m in mask_to_num:
    #   needed = m ^ sig(c) ^ target_mask
    #   if needed in mask_to_num and mask_to_num[needed] != mask_to_num[m] (distinct) and mask_to_num[needed] < c and mask_to_num[m] < c:
    #     found triple
    
    # But this is O(n^2) which might be too slow for large n.
    # Instead, let's be smarter: for each c, we want to know if there exist two distinct 
    # previously-seen numbers a, b such that sig(a) XOR sig(b) = sig(c) XOR target_mask.
    # This is equivalent to: is there a mask m such that both m and m^needed are in mask_to_num?
    # where needed = sig(c) ^ target_mask.
    
    # Precompute: for the set of masks seen so far, maintain a set of all XOR pairs?
    # That's too expensive.
    
    # Alternative: iterate c from 213 to some limit, and for each c do a bounded search.
    # Since we already have a 2-element solution with value 30, we only need 3-element 
    # solutions with max < 240, i.e., c <= 239.
    
    # Let's do a targeted 3-element search for c up to 239
    # For c in 213..239, check if there exist a,b in 211..c-1 with sig(a)^sig(b)^sig(c)^target_mask==0
    
    three_element_search_limit = min(239, max_val)
    mask_to_num2 = {}
    for c in range(213, three_element_search_limit + 1):
        if time.time() > deadline - 5:
            break
        sc = sig_masks[c]
        needed = sc ^ target_mask
        # Check if there exist two distinct a, b < c with sig(a) XOR sig(b) = needed
        # For each mask m in mask_to_num2, check if m^needed is also in mask_to_num2
        # and the two numbers are distinct
        found_triple = False
        for m, num_a in list(mask_to_num2.items()):
            m2 = m ^ needed
            if m2 in mask_to_num2 and mask_to_num2[m2] != num_a:
                num_b = mask_to_num2[m2]
                if num_a < c and num_b < c and num_a != num_b:
                    val = c - 210
                    if val < best_value:
                        best_value = val
                        best_answer = [num_a, num_b, c]
                    found_triple = True
                    break
        if found_triple:
            break
        if sc not in mask_to_num2:
            mask_to_num2[sc] = c

    # Verify the best answer
    if best_answer is not None:
        prod = 210
        for x in best_answer:
            prod *= x
        sqrt_val = math.isqrt(prod)
        if sqrt_val * sqrt_val != prod:
            # Fallback to known valid solution
            best_answer = [840]
            best_value = 630

    if best_answer is None:
        best_answer = [840]
        best_value = 630

    # Ensure all elements are Python int
    return [int(x) for x in best_answer]
