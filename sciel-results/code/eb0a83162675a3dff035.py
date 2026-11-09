import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s wall-clock limit

    n = 210
    max_val = 10210
    max_size = 100

    # 210 = 2 * 3 * 5 * 7 (square-free)
    # We need product of selected * 210 to be a perfect square.
    # This means XOR of parity signatures of selected numbers must equal {2,3,5,7}.

    # Strategy:
    # 1. Find single-element solutions: x = 210 * k^2, x > 210, x <= 10210
    #    Minimum is 210*4 = 840, giving value 630.
    # 2. Find 2-element solutions: sig(a) XOR sig(b) = {2,3,5,7}
    #    For each a, b must have signature sig(a) XOR {2,3,5,7}.
    # 3. Find 3-element solutions if needed.
    # 4. Generalize to more elements using linear algebra over GF(2).

    # Precompute parity signatures for numbers 211 to max_val
    # A parity signature is a frozenset of primes with odd exponent.
    
    def get_parity_signature(x):
        """Return frozenset of primes where x has odd exponent."""
        sig = set()
        temp = x
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    sig.add(p)
            p += 1 if p == 2 else 2
        if temp > 1:
            sig.add(temp)
        return frozenset(sig)

    # Build signature map: for each signature, store the smallest number with that signature
    sig_to_min = {}
    sig_to_all = {}  # signature -> list of numbers (for finding distinct pairs)

    # We'll search from small values upward to find minimal max element
    # First, let's precompute signatures up to a reasonable limit
    # Start with up to 210 + 630 = 840, then extend if needed

    search_limit = 840
    for x in range(211, search_limit + 1):
        sig = get_parity_signature(x)
        if sig not in sig_to_min or x < sig_to_min[sig]:
            sig_to_min[sig] = x
        if sig not in sig_to_all:
            sig_to_all[sig] = []
        sig_to_all[sig].append(x)

    target_sig = frozenset({2, 3, 5, 7})

    best_answer = None
    best_value = float('inf')

    # Check single-element solutions
    for k in range(2, 100):
        x = 210 * k * k
        if x > 210 and x <= max_val:
            val = x - 210
            if val < best_value:
                best_value = val
                best_answer = [x]

    # Try 2-element solutions
    # sig(a) XOR sig(b) = target_sig => sig(b) = sig(a) XOR target_sig
    for sig1, min_num1 in sig_to_min.items():
        sig2 = sig1 ^ target_sig
        if sig2 in sig_to_min:
            min_num2 = sig_to_min[sig2]
            if min_num1 != min_num2:
                max_elem = max(min_num1, min_num2)
                val = max_elem - 210
                if val < best_value:
                    candidate = [min_num1, min_num2]
                    # Verify
                    prod = 210 * min_num1 * min_num2
                    sq = math.isqrt(prod)
                    if sq * sq == prod:
                        best_value = val
                        best_answer = candidate

    # Try 3-element solutions
    # sig(a) XOR sig(b) XOR sig(c) = target_sig
    # => sig(c) = sig(a) XOR sig(b) XOR target_sig
    # To keep it manageable, only consider pairs where both numbers are small
    
    # Collect all signatures and their minimum numbers
    sig_items = list(sig_to_min.items())
    
    for i in range(len(sig_items)):
        if time.time() > deadline - 5:
            break
        sig1, num1 = sig_items[i]
        for j in range(i + 1, len(sig_items)):
            sig2, num2 = sig_items[j]
            if num2 - 210 >= best_value:
                break  # No need to continue with larger num2
            sig3 = sig1 ^ sig2 ^ target_sig
            if sig3 in sig_to_min:
                num3 = sig_to_min[sig3]
                if num3 != num1 and num3 != num2:
                    max_elem = max(num1, num2, num3)
                    val = max_elem - 210
                    if val < best_value:
                        candidate = [num1, num2, num3]
                        prod = 210 * num1 * num2 * num3
                        sq = math.isqrt(prod)
                        if sq * sq == prod:
                            best_value = val
                            best_answer = candidate

    # If we haven't found something better, try extending the search range
    # But first, let's also try a more systematic approach using linear algebra
    
    # For a stronger approach: use Gaussian elimination over GF(2)
    # Each number's signature is a vector in GF(2)^P where P is the set of relevant primes
    # We want a subset whose XOR equals target_sig
    
    # Let's identify all distinct signatures and build a matrix
    # But this could be expensive. Instead, let's try a greedy/iterative approach.
    
    # Alternative: for each possible max value M from 211 upward, check if there exists
    # a valid subset using only numbers in (210, M]. The first such M gives the answer.
    
    # This is more systematic but potentially slower. Let's try it with a limit.
    
    if best_value > 100:  # If we haven't found something good, try systematic search
        # Systematic search: for M from 211 to 210 + best_value - 1, check feasibility
        # Use the signature approach: collect all signatures for numbers in (210, M]
        # and check if target_sig is in their span over GF(2)
        
        # To make this tractable, limit the number of distinct signatures
        # and use a simple subset-sum-like check for small sets
        
        # Actually, let's just extend our sig_to_min and try more combinations
        extended_limit = min(max_val, 210 + best_value - 1)
        for x in range(search_limit + 1, extended_limit + 1):
            if time.time() > deadline - 5:
                break
            sig = get_parity_signature(x)
            if sig not in sig_to_min or x < sig_to_min[sig]:
                sig_to_min[sig] = x

        # Retry 2-element and 3-element searches with extended data
        sig_items2 = list(sig_to_min.items())
        
        # 2-element
        for sig1, min_num1 in sig_items2:
            sig2 = sig1 ^ target_sig
            if sig2 in sig_to_min:
                min_num2 = sig_to_min[sig2]
                if min_num1 != min_num2:
                    max_elem = max(min_num1, min_num2)
                    val = max_elem - 210
                    if val < best_value:
                        prod = 210 * min_num1 * min_num2
                        sq = math.isqrt(prod)
                        if sq * sq == prod:
                            best_value = val
                            best_answer = [min_num1, min_num2]

        # 3-element (limited)
        for i in range(len(sig_items2)):
            if time.time() > deadline - 5:
                break
            sig1, num1 = sig_items2[i]
            for j in range(i + 1, len(sig_items2)):
                sig2, num2 = sig_items2[j]
                if num2 - 210 >= best_value:
                    break
                sig3 = sig1 ^ sig2 ^ target_sig
                if sig3 in sig_to_min:
                    num3 = sig_to_min[sig3]
                    if num3 != num1 and num3 != num2:
                        max_elem = max(num1, num2, num3)
                        val = max_elem - 210
                        if val < best_value:
                            prod = 210 * num1 * num2 * num3
                            sq = math.isqrt(prod)
                            if sq * sq == prod:
                                best_value = val
                                best_answer = [num1, num2, num3]

    # Final verification and return
    if best_answer is None:
        # Fallback: single element 840
        best_answer = [840]
    else:
        # Verify the answer
        prod = 210
        for x in best_answer:
            prod *= x
        sq = math.isqrt(prod)
        if sq * sq != prod:
            # Invalid, use fallback
            best_answer = [840]
            best_value = 630

    # Ensure all elements are Python ints
    result = [int(x) for x in best_answer]
    
    # Final checks
    assert isinstance(result, list)
    assert 1 <= len(result) <= 100
    assert len(set(result)) == len(result)
    for x in result:
        assert isinstance(x, int) and not isinstance(x, bool)
        assert 210 < x <= 10210
    
    return result
