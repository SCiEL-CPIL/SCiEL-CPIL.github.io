import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    n = 210
    max_val = 10210
    max_size = 100

    # SPF sieve
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_sig(x):
        sig = set()
        temp = x
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                sig.add(p)
        return frozenset(sig)

    target = frozenset({2, 3, 5, 7})

    sig_to_numbers = {}
    sigs = [None] * (max_val + 1)
    for x in range(211, max_val + 1):
        s = get_sig(x)
        sigs[x] = s
        if s not in sig_to_numbers:
            sig_to_numbers[s] = []
        sig_to_numbers[s].append(x)

    best_value = float('inf')
    best_answer = None

    # 1-element
    for k in range(1, 100):
        x = 210 * k * k
        if x > 210 and x <= max_val:
            val = x - 210
            if val < best_value:
                best_value = val
                best_answer = [x]
        else:
            break

    # 2-element
    processed_sigs = set()
    for s1, nums1 in sig_to_numbers.items():
        if s1 in processed_sigs:
            continue
        s2 = s1 ^ target
        if s2 not in sig_to_numbers:
            continue
        processed_sigs.add(s1)
        processed_sigs.add(s2)

        nums2 = sig_to_numbers[s2]

        if s1 == s2:
            if len(nums1) >= 2:
                a, b = nums1[0], nums1[1]
                max_elem = max(a, b)
                val = max_elem - 210
                if val < best_value:
                    best_value = val
                    best_answer = [a, b]
        else:
            a = nums1[0]
            b = nums2[0]
            if a != b:
                max_elem = max(a, b)
                val = max_elem - 210
                if val < best_value:
                    best_value = val
                    best_answer = [a, b]
            else:
                if len(nums2) >= 2:
                    b2 = nums2[1]
                    max_elem = max(a, b2)
                    val = max_elem - 210
                    if val < best_value:
                        best_value = val
                        best_answer = [a, b2]
                if len(nums1) >= 2:
                    a2 = nums1[1]
                    max_elem = max(a2, b)
                    val = max_elem - 210
                    if val < best_value:
                        best_value = val
                        best_answer = [a2, b]

    # 3-element search
    # We want to find a, b, c such that sig(a)^sig(b)^sig(c) == target
    # and max(a,b,c) < best_value + 210
    # Let limit = best_value + 210
    # We iterate a, b < limit
    # For each pair, find smallest c < limit, c!=a, c!=b with sig(c) == sig(a)^sig(b)^target
    
    limit = int(best_value) + 210
    # If best_value is inf, limit is huge, but we capped max_val.
    # Actually, if no 1 or 2 elem found, best_value is inf. But 1 elem always exists (840).
    # So limit is at most 840.
    # We search for max_elem < limit.
    
    best_3_value = float('inf')
    best_3_answer = None
    
    # To speed up, we can iterate a, b in increasing order of max(a,b)?
    # Or just iterate a from 211 to limit-1, b from a+1 to limit-1.
    # Prune if max(a,b) >= best_3_value + 210? No, max(a,b,c) could be smaller if c is small?
    # No, max(a,b,c) >= max(a,b). So if max(a,b) >= current_best_3_max, we can skip?
    # Yes.
    
    current_best_3_max = limit
    
    for a in range(211, limit):
        if a >= current_best_3_max:
            break
        sig_a = sigs[a]
        for b in range(a + 1, limit):
            if b >= current_best_3_max:
                break
            sig_b = sigs[b]
            req_sig = sig_a ^ sig_b ^ target
            
            if req_sig not in sig_to_numbers:
                continue
            
            candidates = sig_to_numbers[req_sig]
            # Find smallest c in candidates such that c < current_best_3_max, c != a, c != b
            # candidates is sorted
            # We can iterate or binary search.
            # Since we want smallest c, we iterate from start.
            # But we can stop if c >= current_best_3_max.
            
            found_c = None
            for c in candidates:
                if c >= current_best_3_max:
                    break
                if c == a or c == b:
                    continue
                found_c = c
                break
            
            if found_c is not None:
                max_elem = max(a, b, found_c)
                if max_elem < current_best_3_max:
                    current_best_3_max = max_elem
                    val = max_elem - 210
                    if val < best_3_value:
                        best_3_value = val
                        best_3_answer = [a, b, found_c]
                    # We can update limit? No, limit is fixed for the search space relative to 2-elem best.
                    # But we can prune b loop if b >= current_best_3_max.
                    # And a loop if a >= current_best_3_max.
    
    if best_3_answer is not None:
        if best_3_value < best_value:
            best_value = best_3_value
            best_answer = best_3_answer

    if best_answer is not None:
        prod = 210
        for x in best_answer:
            prod *= x
        sqrt_val = math.isqrt(prod)
        if sqrt_val * sqrt_val != prod:
            best_answer = [840]
            best_value = 630
        return [int(x) for x in best_answer]

    return [840]
