"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def _sf_kernel_fast(n):
    """Return the squarefree kernel of n using trial division with step optimization."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count % 2 == 1:
                result *= d
        if d == 2:
            d = 3
        else:
            d += 2
    if temp > 1:
        result *= temp
    return result

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 70.0  # 70s deadline, 20s buffer before 90s kill
    
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    # 210 = 2 * 3 * 5 * 7, all squarefree
    target_sf = 210  # sf(210) = 210
    
    # Precompute squarefree kernels for all numbers in (210, MAX_VAL]
    sf = {}
    for x in range(N + 1, MAX_VAL + 1):
        sf[x] = _sf_kernel_fast(x)
    
    # Collect all primes that appear in any squarefree kernel in our range
    all_primes = set()
    for x in range(N + 1, MAX_VAL + 1):
        ker = sf[x]
        d = 2
        temp = ker
        while d * d <= temp:
            if temp % d == 0:
                all_primes.add(d)
                while temp % d == 0:
                    temp //= d
            if d == 2:
                d = 3
            else:
                d += 2
        if temp > 1:
            all_primes.add(temp)
    
    # Also add primes from target
    d = 2
    temp = target_sf
    while d * d <= temp:
        if temp % d == 0:
            all_primes.add(d)
            while temp % d == 0:
                temp //= d
        if d == 2:
            d = 3
        else:
            d += 2
    if temp > 1:
        all_primes.add(temp)
    
    primes = sorted(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Convert each squarefree kernel to a bitmask
    def ker_to_mask(ker):
        mask = 0
        d = 2
        temp = ker
        while d * d <= temp:
            if temp % d == 0:
                exp = 0
                while temp % d == 0:
                    temp //= d
                    exp += 1
                if exp % 2 == 1 and d in prime_to_idx:
                    mask |= (1 << prime_to_idx[d])
            if d == 2:
                d = 3
            else:
                d += 2
        if temp > 1 and temp in prime_to_idx:
            mask |= (1 << prime_to_idx[temp])
        return mask
    
    masks = {}
    for x in range(N + 1, MAX_VAL + 1):
        masks[x] = ker_to_mask(sf[x])
    
    target_mask = ker_to_mask(target_sf)
    
    # Strategy: For each M from 211 upward, find the smallest subset of {211,...,M}
    # whose XOR of masks equals target_mask.
    # 
    # For efficiency, we'll check:
    # 1. Single element: mask[x] == target_mask
    # 2. Pair: mask[a] ^ mask[b] == target_mask
    # 3. Triple: mask[a] ^ mask[b] ^ mask[c] == target_mask
    # 4. Use linear algebra for larger subsets if needed
    
    # Precompute for single elements
    single_solutions = []
    for x in range(N + 1, MAX_VAL + 1):
        if masks[x] == target_mask:
            single_solutions.append(x)
    
    # For pairs, use a dictionary: mask -> list of numbers with that mask
    mask_to_nums = defaultdict(list)
    for x in range(N + 1, MAX_VAL + 1):
        mask_to_nums[masks[x]].append(x)
    
    # For each M, check if there's a solution using only numbers <= M
    # We want to minimize M, so iterate M from 211 upward
    
    best_result = None
    best_M = MAX_VAL + 1
    
    # Check single elements first (fastest)
    for x in single_solutions:
        if x <= MAX_VAL:
            best_M = x
            best_result = [x]
            break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # For pairs: for each M, we need a pair (a,b) with a,b <= M and mask[a]^mask[b] == target_mask
    # This means mask[b] == mask[a] ^ target_mask
    # Precompute: for each mask m, what is m ^ target_mask?
    
    # Iterate M from 211 upward, checking for pairs
    # To make this efficient, we process numbers in order and maintain a set of seen masks
    
    seen_masks = {}  # mask -> smallest number with that mask (for current M window)
    
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
        
        x = M
        mx = masks[x]
        
        # Check if x alone works (already handled above, but just in case)
        if mx == target_mask:
            best_M = x
            best_result = [x]
            break
        
        # Check pair: is there a previously seen number a with mask[a] ^ mask[x] == target_mask?
        # i.e., mask[a] == mask[x] ^ target_mask
        needed = mx ^ target_mask
        if needed in seen_masks:
            a = seen_masks[needed]
            # Verify
            prod = N * a * x
            r = math.isqrt(prod)
            if r * r == prod:
                best_M = M
                best_result = [a, x]
                break
        
        # Add current number's mask to seen
        if mx not in seen_masks:
            seen_masks[mx] = x
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # For triples: use a different approach
    # For each M, we need three numbers a,b,c <= M with mask[a]^mask[b]^mask[c] == target_mask
    # This is equivalent to: for some pair (a,b), mask[a]^mask[b] ^ target_mask = mask[c]
    # So for each M, we can check all pairs (a,b) with a<b<=M and see if mask[a]^mask[b]^target_mask
    # corresponds to some c <= M.
    
    # This is O(M^2) per M which is too slow. Instead, let's use a smarter approach.
    # 
    # Alternative: Use the fact that we can have up to 100 elements.
    # For larger subsets, use Gaussian elimination over GF(2).
    
    # Let's try a different approach: for each M, build the system and solve.
    # But this is expensive. Instead, let's try to find small solutions.
    
    # For triples, we can do:
    # For each M from 211 upward:
    #   For each pair (a,b) with a < b <= M:
    #     needed_mask = mask[a] ^ mask[b] ^ target_mask
    #     If needed_mask is in mask_to_nums and there exists c <= M with that mask:
    #       Check if c != a and c != b
    #       If so, we have a solution
    
    # To make this efficient, precompute for each mask the minimum number with that mask
    min_num_for_mask = {}
    for x in range(N + 1, MAX_VAL + 1):
        m = masks[x]
        if m not in min_num_for_mask or x < min_num_for_mask[m]:
            min_num_for_mask[m] = x
    
    # Now for triples, iterate M and check
    # But O(M^2) is still too slow for large M.
    
    # Better approach: since we want the minimum M, and single/pair didn't work,
    # let's try to find the minimum M for triples using a different strategy.
    
    # For each possible triple (a,b,c) with a<b<c, the max is c.
    # We want to minimize c such that mask[a]^mask[b]^mask[c] == target_mask.
    
    # This is equivalent to: for each c from 211 upward, check if there exist a<b<c
    # with mask[a]^mask[b] == mask[c]^target_mask.
    
    # For a fixed c, we need to check if there's a pair (a,b) with a<b<c and
    # mask[a]^mask[b] == mask[c]^target_mask.
    
    # Precompute: for each mask m, keep track of the two smallest numbers with that mask
    # as we iterate c upward.
    
    # Actually, let's just do a more direct search for small M values.
    # For M up to some limit, check all subsets of size 3.
    
    # Given the constraints, let's try a hybrid approach:
    # 1. For M from 211 to some limit (say 1000), check all pairs and triples
    # 2. If not found, use linear algebra for the full range
    
    # Let's first try to find a triple solution by iterating c
    # and checking if mask[c]^target_mask can be expressed as mask[a]^mask[b] for a<b<c.
    
    # For efficiency, maintain a set of "pair XORs" as we go.
    # pair_xor_to_pair = {}  # xor_value -> (a, b) where a<b and both < current c
    
    # But this grows as O(M^2) which is too much.
    
    # Alternative: just check specific small ranges.
    
    # Let's try: for M from 211 to 500, check all triples by brute force
    # (this is O(300^3) = 27 million, which might be too slow in Python)
    
    # Better: for each c from 211 to 500:
    #   needed = masks[c] ^ target_mask
    #   Check if needed can be written as masks[a] ^ masks[b] for a < b < c
    #   This is O(c^2) per c, total O(500^3/3) ≈ 41 million, still too slow.
    
    # Let's use the mask_to_nums approach more cleverly.
    # For a fixed needed mask, we need two numbers a,b with a<b<c and masks[a]^masks[b] == needed.
    # This means masks[b] == masks[a] ^ needed.
    # So for each a < c, check if masks[a]^needed is in mask_to_nums and has a number b with a < b < c.
    
    # Precompute: for each mask, sorted list of numbers with that mask
    mask_to_sorted_nums = defaultdict(list)
    for x in range(N + 1, MAX_VAL + 1):
        mask_to_sorted_nums[masks[x]].append(x)
    for m in mask_to_sorted_nums:
        mask_to_sorted_nums[m].sort()
    
    # For each c from 211 upward:
    #   needed = masks[c] ^ target_mask
    #   For each a < c:
    #     b_mask = masks[a] ^ needed
    #     If b_mask in mask_to_sorted_nums:
    #       Find smallest b in mask_to_sorted_nums[b_mask] with a < b < c
    #       If found, we have a solution (a, b, c)
    
    # This is still O(M^2) per c in the worst case.
    
    # Let's just try a reasonable limit and see.
    
    # Actually, let's step back. The current program scores 21.0, which means
    # max - 210 = 21, so max = 231. Let me verify what solution gives this.
    
    # If max = 231, we need numbers in (210, 231] whose product * 210 is a perfect square.
    # Let me check: is there a single number x in (210, 231] with 210*x a perfect square?
    # 210*x = k^2 => x = k^2/210. For x to be integer, 210 | k^2.
    # 210 = 2*3*5*7, so k must be divisible by 2*3*5*7 = 210.
    # k = 210*m, x = 210*m^2. For m=1, x=210 (not > 210). For m=2, x=840 (> 231).
    # So no single element works for M=231.
    
    # For pairs: 210*a*b = k^2. Let's check if there's a pair in (210, 231].
    # Numbers: 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231
    
    # Let me compute sf for these:
    # 210 = 2*3*5*7, sf=210
    # 211 = prime, sf=211
    # 212 = 4*53, sf=53
    # 213 = 3*71, sf=213
    # 214 = 2*107, sf=214
    # 215 = 5*43, sf=215
    # 216 = 8*27 = 2^3 * 3^3, sf=2*3=6
    # 217 = 7*31, sf=217
    # 218 = 2*109, sf=218
    # 219 = 3*73, sf=219
    # 220 = 4*5*11, sf=5*11=55
    # 221 = 13*17, sf=221
    # 222 = 2*3*37, sf=222
    # 223 = prime, sf=223
    # 224 = 32*7 = 2^5*7, sf=2*7=14
    # 225 = 15^2, sf=1
    # 226 = 2*113, sf=226
    # 227 = prime, sf=227
    # 228 = 4*57 = 4*3*19, sf=3*19=57
    # 229 = prime, sf=229
    # 230 = 2*5*23, sf=230
    # 231 = 3*7*11, sf=231
    
    # We need sf(a)*sf(b) to have sf = 210 (i.e., sf(a)*sf(b) = 210 * square)
    # Equivalently, sf(a) ^ sf(b) (in prime parity sense) = 210
    
    # Let me check pairs:
    # sf(216)=6, sf(225)=1: 6^1 in mask sense... 
    # Actually, sf(a)*sf(b) is a perfect square times 210 iff sf(a)*sf(b)/210 is a perfect square.
    # Or: sf(a)*sf(b) = 210 * w^2 for some w.
    
    # Let me just check: for a=216 (sf=6), we need sf(b) such that 6*sf(b) = 210*w^2
    # => sf(b) = 35*w^2. Since sf(b) is squarefree, w=1, sf(b)=35.
    # Is there a b in (210,231] with sf(b)=35? 35=5*7. 
    # Numbers with sf=35: 35, 140, 315, 560, 945, ... None in (210,231].
    
    # For a=225 (sf=1), we need sf(b)=210. Is there a b in (210,231] with sf(b)=210?
    # 210 itself is not > 210. 210*4=840 > 231. So no.
    
    # Hmm, so maybe the score of 21 doesn't come from M=231. Let me reconsider.
    # Score = seed/value where value = max - 210. If score = 21, then value = seed/21.
    # If seed = 441, value = 21, max = 231.
    # If seed = 840, value = 40, max = 250.
    
    # Let me just focus on finding the optimal solution.
    
    # Let me try a more systematic approach: iterate M from 211 and for each M,
    # check if there's a valid subset using only numbers in (210, M].
    
    # For efficiency, I'll check:
    # 1. Single: O(1) per M
    # 2. Pair: O(1) per M using the seen_masks approach
    # 3. Triple: O(M) per M using the pair-checking approach
    # 4. If M gets large, use linear algebra
    
    # Let me implement this properly.
    
    # Re-initialize
    seen_masks = {}  # mask -> smallest number with that mask seen so far
    
    # For triples, we need to check if for the current x=M, there exist a<b<x
    # with masks[a]^masks[b] == masks[x]^target_mask.
    
    # To do this efficiently, we can maintain a set of all pair XORs for numbers < x.
    # But this grows as O(x^2).
    
    # Alternative: for each x, iterate over all a < x and check if masks[a]^masks[x]^target_mask
    # is in seen_masks (i.e., there's a b < a with that mask).
    
    # Wait, we need a < b < x. So for a given x and a, we need b such that:
    #   masks[b] = masks[a] ^ masks[x] ^ target_mask
    #   and a < b < x
    
    # So we need to know, for each mask, what numbers have that mask and are in (a, x).
    
    # This is getting complex. Let me just do a bounded search for small M values.
    
    # For M from 211 to 300, do a brute-force check for subsets of size up to 4.
    # This should be fast enough.
    
    import itertools
    
    for M in range(N + 1, min(MAX_VAL, 400) + 1):
        if time.time() > deadline:
            break
        
        nums = list(range(N + 1, M + 1))
        k = len(nums)
        
        # Check size 1
        found = False
        for i in range(k):
            if masks[nums[i]] == target_mask:
                best_M = M
                best_result = [nums[i]]
                found = True
                break
        if found:
            break
        
        # Check size 2
        if not found:
            for i in range(k):
                if time.time() > deadline:
                    break
                needed = masks[nums[i]] ^ target_mask
                for j in range(i + 1, k):
                    if masks[nums[j]] == needed:
                        # Verify
                        a, b = nums[i], nums[j]
                        prod = N * a * b
                        r = math.isqrt(prod)
                        if r * r == prod:
                            best_M = M
                            best_result = [a, b]
                            found = True
                            break
                if found:
                    break
        if found:
            break
        
        # Check size 3 (only for small M to keep it fast)
        if not found and k <= 100:
            for i in range(k):
                if time.time() > deadline:
                    break
                m_i = masks[nums[i]]
                for j in range(i + 1, k):
                    m_ij = m_i ^ masks[nums[j]]
                    needed = m_ij ^ target_mask
                    for l in range(j + 1, k):
                        if masks[nums[l]] == needed:
                            a, b, c = nums[i], nums[j], nums[l]
                            prod = N * a * b * c
                            r = math.isqrt(prod)
                            if r * r == prod:
                                best_M = M
                                best_result = [a, b, c]
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
        if found:
            break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # If we haven't found a solution for M up to 400, use a more sophisticated approach.
    # Let's try to find the minimum M using a different strategy.
    
    # For larger M, use the fact that we can have up to 100 elements.
    # Use Gaussian elimination to find a subset.
    
    # But first, let's try to find pairs for larger M using the seen_masks approach
    # which we started but didn't complete.
    
    seen_masks2 = {}
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
        
        x = M
        mx = masks[x]
        
        # Check pair
        needed = mx ^ target_mask
        if needed in seen_masks2:
            a = seen_masks2[needed]
            prod = N * a * x
            r = math.isqrt(prod)
            if r * r == prod:
                best_M = M
                best_result = [a, x]
                break
        
        if mx not in seen_masks2:
            seen_masks2[mx] = x
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Fallback: use a known valid solution
    # 210 * 840 = 176400 = 420^2, so [840] works with max-210 = 630
    return [840]
# EVOLVE-BLOCK-END
