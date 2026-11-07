"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    deadline = time.time() + 55  # safety margin before 60s/90s kill
    N = 210
    MAX_VAL = 10210
    MIN_VAL = 211
    
    # Factorize 210: 2*3*5*7
    # We need product * 210 to be a perfect square
    # So product must have odd exponents for primes 2,3,5,7 and even for all others
    # Equivalently: parity vector of product XOR parity vector of 210 = 0
    # i.e., parity vector of product = parity vector of 210
    
    # Get all primes up to MAX_VAL for factorization
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    primes = sieve_primes(MAX_VAL)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    nprimes = len(primes)
    
    # Target parity vector: 210 = 2^1 * 3^1 * 5^1 * 7^1
    target = [0] * nprimes
    for p in [2, 3, 5, 7]:
        if p in prime_to_idx:
            target[prime_to_idx[p]] = 1
    
    def get_parity(n):
        """Return parity vector (as frozenset of prime indices with odd exponent) for n."""
        parity = set()
        for p in primes:
            if p * p > n:
                break
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count % 2 == 1:
                parity.add(prime_to_idx[p])
        if n > 1:
            parity.add(prime_to_idx[n])
        return frozenset(parity)
    
    # For each candidate number, compute its parity vector
    # We want a subset S such that XOR of parities = target
    # This is a linear algebra problem over GF(2)
    
    # Strategy: iterate M from MIN_VAL upward, maintain all numbers in [MIN_VAL, M]
    # Use Gaussian elimination to find if target is in the span
    # Track the actual subset using back-references
    
    # Represent vectors as tuples of 0/1 for the primes
    # But nprimes can be large (~1200), so use bitsets or sparse reps
    
    # Better: use a dictionary mapping frozenset -> (number, parent_info)
    # Actually, let's use a standard GF(2) Gaussian elimination with tracking
    
    # For efficiency, represent each vector as a tuple of indices where bit is 1
    # Use a dict: vector_key -> (number, list_of_numbers_in_combo)
    
    # Actually, the cleanest approach: 
    # Maintain a basis in row-echelon form over GF(2)
    # Each basis row has a leading position and a vector
    # Also track which original numbers were used
    
    # Let me use a different approach: 
    # For each M, add number M to our set
    # Try to reduce M against existing basis
    # If it reduces to zero, M is dependent
    # If it reduces to a new pivot, add it
    # Check if target can be formed
    
    # But we need the ACTUAL subset, not just existence
    
    # Let me use: for each vector in the span, store one representation
    # But the span can be huge...
    
    # Alternative: since we want MINIMAL max, and numbers are small,
    # the solution likely uses few numbers. Let's try:
    # 1. Single number: need n * 210 = perfect square => n = 210 * k^2 / 210... 
    #    n * 210 = m^2 => n = m^2 / 210. So 210 | m^2, meaning 210's squarefree part divides m.
    #    210 = 2*3*5*7, all to power 1. So m must be divisible by 2*3*5*7 = 210.
    #    m = 210*k, n = 210*k^2. Need 211 <= 210*k^2 <= 10210.
    #    k=1: n=210 (not > 210)
    #    k=2: n=840. 840*210 = 176400 = 420^2. Yes!
    #    So [840] works with max-210 = 630.
    
    # But we can do better with more numbers.
    # 2. Two numbers a, b: a*b*210 = perfect square
    #    parity(a) XOR parity(b) = target
    #    So parity(b) = parity(a) XOR target
    #    For each a, compute needed parity for b, check if any b in range has that parity.
    
    # Let's precompute parity vectors for all candidates
    # Then for two-element search: for each a, look up if (parity(a) XOR target) exists
    
    # This is O(n^2) naive, but with a hash map it's O(n)
    
    # Precompute all parities
    parities = {}
    for n in range(MIN_VAL, MAX_VAL + 1):
        parities[n] = get_parity(n)
    
    # Build reverse map: parity -> list of numbers with that parity
    from collections import defaultdict
    parity_to_nums = defaultdict(list)
    for n in range(MIN_VAL, MAX_VAL + 1):
        parity_to_nums[parities[n]].append(n)
    
    # Target as frozenset
    target_fs = frozenset(i for i, v in enumerate(target) if v == 1)
    
    # Check 1-element solutions
    best_result = None
    best_raw = float('inf')
    
    # 1-element: n * 210 is perfect square
    # n = 210 * k^2 where 210*k^2 > 210 and <= 10210
    # k^2 > 1, k^2 <= 10210/210 ≈ 48.6, so k = 2,3,4,5,6,7
    for k in range(2, 8):
        n = 210 * k * k
        if MIN_VAL <= n <= MAX_VAL:
            # Verify
            prod = n * 210
            s = isqrt(prod)
            if s * s == prod:
                raw = n - 210
                if raw < best_raw:
                    best_raw = raw
                    best_result = [n]
    
    # 2-element: a * b * 210 is perfect square, a != b
    # parity(a) XOR parity(b) = target
    # For each a, needed_b_parity = parity(a) XOR target
    # Find smallest b > a with that parity (to minimize max)
    
    # Iterate M from small to large, check if any pair (a,b) with max(a,b)=M works
    # Actually, just iterate all a, find valid b, track min max
    
    for a in range(MIN_VAL, MAX_VAL + 1):
        pa = parities[a]
        # Compute XOR with target
        # pa is frozenset, target_fs is frozenset
        needed = pa.symmetric_difference(target_fs)
        if needed in parity_to_nums:
            for b in parity_to_nums[needed]:
                if b != a and b >= MIN_VAL and b <= MAX_VAL:
                    raw = max(a, b) - 210
                    if raw < best_raw:
                        best_raw = raw
                        best_result = [a, b]
    
    # 3-element: a*b*c*210 is perfect square
    # parity(a) XOR parity(b) XOR parity(c) = target
    # For each pair (a,b), needed_c_parity = parity(a) XOR parity(b) XOR target
    # This is O(n^2) which is ~10^8 for n~10000, too slow
    # But we can be smarter: iterate M, and for each M as the max, 
    # check if there exist a,b < M such that parity(a) XOR parity(b) = parity(M) XOR target
    
    # Build a set of parities seen so far as we iterate M upward
    # For each M, we need: exists a,b in [MIN_VAL, M-1] with parity(a) XOR parity(b) = parity(M) XOR target
    
    # This is equivalent to: for each a in seen, check if parity(a) XOR (parity(M) XOR target) is also in seen
    # That's O(n) per M, total O(n^2) ~ 10^8, might be tight in 55s
    
    # Let's try it with time checks
    seen_parities = set()
    # We need to track not just parities but also which numbers have them
    # For 3-element, we need distinct a, b, c
    
    # Actually, let me just do a more targeted search for small M values
    # Since we already have a 2-element solution, let's see if 3-element can beat it
    
    # Current best_raw is from 2-element. Let's see what it is.
    # If best_raw is small, 3-element might not improve much.
    
    # Let's try 3-element with a limit
    if time.time() < deadline - 5:
        # For 3-element: iterate c from MIN_VAL to MAX_VAL
        # Maintain a dict: parity -> list of numbers (up to 2) to allow distinctness
        parity_counts = defaultdict(list)  # parity -> [num1, num2, ...]
        
        for c in range(MIN_VAL, MAX_VAL + 1):
            if time.time() > deadline - 3:
                break
            pc = parities[c]
            needed_ab_parity = pc.symmetric_difference(target_fs)
            # We need a, b in seen with parity(a) XOR parity(b) = needed_ab_parity
            # For each a in seen, check if parity(a) XOR needed_ab_parity is in seen
            # But we need to be careful about distinctness
            
            # More efficient: maintain a set of all parities seen, and for each, 
            # we can check if the complementary parity exists
            # But we need actual numbers for distinctness
            
            # Let's just check: for each a in a limited set of seen numbers
            # Actually, this is getting complex. Let me try a different approach.
            
            # For 3-element with max = c, we need a,b < c (or <= c but distinct)
            # Let's just try small c values since we want to minimize max
            
            if c - 210 >= best_raw:
                break  # Can't improve
            
            # For this c, find if there exist distinct a,b in [MIN_VAL, c] 
            # (with a,b != c if c is used, but let's allow a or b = c for now, 
            #  actually they must be distinct from each other and from c)
            # Hmm, let's require a < b < c or a < c, b < c with a != b
            
            # Simplify: look for a, b in [MIN_VAL, c-1] with parity(a) XOR parity(b) = needed_ab_parity
            # We can precompute as we go
            
            # This inner loop is expensive. Let me use a set of parities.
            # For each parity p in seen_parities, check if p XOR needed_ab_parity is also in seen_parities
            # If yes, we need to verify there are distinct numbers for both parities (or same parity with 2+ numbers)
            
            # Let's maintain: parity -> count of numbers seen with that parity
            # And also track specific numbers for reconstruction
            
            # This is getting complicated. Let me just do a direct search for small improvements.
            pass
    
    # Let me try a cleaner 3-element approach with time limit
    if time.time() < deadline - 10 and best_result is not None:
        # Rebuild parity info
        # For 3-element: we want min max(a,b,c) such that parity(a)^parity(b)^parity(c) = target
        # Iterate max_val from MIN_VAL upward
        # For each max_val, check if there's a valid triple with max = max_val
        
        # Precompute parities for all
        # Already have parities dict
        
        # Build: for each parity, the sorted list of numbers with that parity
        # Already have parity_to_nums
        
        # For a triple with max = M:
        # Case 1: M is the largest, a,b < M
        # Case 2: M appears, and one of a,b = M (but they must be distinct, so at most one of a,b can be M)
        # Actually all three must be distinct, so if M is in the triple, the other two are < M
        
        # For max = M:
        # We need distinct a,b,c in [MIN_VAL, M] with max = M
        # So at least one of a,b,c equals M
        # WLOG, let c = M. Then we need a,b in [MIN_VAL, M-1], a != b, 
        # with parity(a) XOR parity(b) = parity(M) XOR target
        
        # For each M, we need to check if there exist a,b in [MIN_VAL, M-1] with that property
        
        # Maintain as we iterate M: a set of parities of numbers in [MIN_VAL, M-1]
        # And for each parity, how many numbers (to handle the case where a and b have the same parity)
        
        seen_parity_count = defaultdict(int)
        seen_parity_nums = defaultdict(list)  # Keep up to 2 for reconstruction
        
        for M in range(MIN_VAL, MAX_VAL + 1):
            if time.time() > deadline - 5:
                break
            if M - 210 >= best_raw:
                break
            
            pM = parities[M]
            needed = pM.symmetric_difference(target_fs)
            
            # Check if there exist a, b in seen with parity(a) XOR parity(b) = needed
            # For each parity p in seen_parity_count:
            #   complement = p XOR needed
            #   If complement == p: need count[p] >= 2
            #   Else: need complement in seen_parity_count and count[complement] >= 1
            
            found = False
            for p in list(seen_parity_count.keys()):
                comp = p.symmetric_difference(needed)
                if comp == p:
                    if seen_parity_count[p] >= 2:
                        # Find two distinct numbers with parity p
                        if len(seen_parity_nums[p]) >= 2:
                            a, b = seen_parity_nums[p][0], seen_parity_nums[p][1]
                            found = True
                            break
                else:
                    if comp in seen_parity_count and seen_parity_count[comp] >= 1:
                        if len(seen_parity_nums[comp]) >= 1 and len(seen_parity_nums[p]) >= 1:
                            a, b = seen_parity_nums[p][0], seen_parity_nums[comp][0]
                            if a != b:
                                found = True
                                break
                            elif len(seen_parity_nums[p]) >= 2:
                                a, b = seen_parity_nums[p][1], seen_parity_nums[comp][0]
                                found = True
                                break
                            elif len(seen_parity_nums[comp]) >= 2:
                                a, b = seen_parity_nums[p][0], seen_parity_nums[comp][1]
                                found = True
                                break
            
            if found:
                triple = [a, b, M]
                # Verify
                prod = triple[0] * triple[1] * triple[2] * 210
                s = isqrt(prod)
                if s * s == prod:
                    raw = M - 210
                    if raw < best_raw:
                        best_raw = raw
                        best_result = triple
                break  # No need to check larger M since we're iterating upward
            
            # Add M to seen
            seen_parity_count[pM] += 1
            if len(seen_parity_nums[pM]) < 2:
                seen_parity_nums[pM].append(M)
    
    # Verify final result
    if best_result is None:
        # Fallback: should not happen since [840] works
        best_result = [840]
    
    # Final verification
    prod = 210
    for x in best_result:
        prod *= x
    s = isqrt(prod)
    if s * s != prod:
        # Something went wrong, fallback
        best_result = [840]
    
    # Ensure types are correct
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
