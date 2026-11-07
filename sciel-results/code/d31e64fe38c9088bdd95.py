"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def _sf_kernel(n):
    """Return the squarefree kernel of n."""
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
        d += 1
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
    deadline = start_time + 80.0
    
    n = 210
    max_val = 10210
    max_size = 100
    
    # Precompute squarefree kernels for all numbers in (210, 10210]
    sf = {}
    for x in range(n + 1, max_val + 1):
        sf[x] = _sf_kernel(x)
    
    # Target: we need product * 210 to be a perfect square
    # So XOR of sf(selected) must equal sf(210)
    target_sf = _sf_kernel(n)  # 210 = 2*3*5*7, all to power 1, so sf(210) = 210
    
    # Strategy: For each possible max value M from 211 upward:
    # 1. Check if there's a single number x in (210, M] with sf(x) == target_sf
    #    (since x * 210 needs to be a square, sf(x) must equal sf(210) = 210)
    # 2. Check if there's a pair (a, b) in (210, M] with sf(a) * sf(b) == target_sf
    #    (i.e., sf(a) ^ sf(b) == target_sf in terms of prime parity)
    #    Actually, we need sf(a) * sf(b) / gcd(sf(a)*sf(b), something)... 
    #    Wait, the condition is that the product of all selected numbers times 210 is a perfect square.
    #    This means the squarefree part of (210 * prod(selected)) must be 1.
    #    Equivalently, the squarefree part of prod(selected) must equal the squarefree part of 210
    #    (since sf(210 * P) = sf(210) * sf(P) / gcd(sf(210), sf(P))... no that's not right either)
    
    # Let me think more carefully.
    # 210 * a * b * ... * z is a perfect square iff for every prime p,
    # the total exponent of p in 210 * a * b * ... * z is even.
    # 
    # Let v_p(x) be the exponent of prime p in x.
    # We need: v_p(210) + sum of v_p(x_i) is even for all p.
    # 
    # This is equivalent to: the squarefree kernel of (210 * prod(x_i)) is 1.
    # 
    # The squarefree kernel of a product is the product of squarefree kernels 
    # with exponents mod 2. So sf(210 * prod(x_i)) = sf(210) * sf(x_1) * ... * sf(x_k) mod squares.
    # More precisely, if we write each number as s * t^2 where s is squarefree,
    # then the product has squarefree part = sf(210) * sf(x_1) * ... * sf(x_k) where we
    # take the product and reduce mod squares (i.e., cancel pairs of identical primes).
    #
    # So we need: sf(210) * sf(x_1) * ... * sf(x_k) to be a perfect square.
    # This means sf(210) * sf(x_1) * ... * sf(x_k) = (some integer)^2.
    # Since all factors are squarefree, this product is a perfect square iff 
    # each prime appears an even number of times across all the squarefree kernels.
    #
    # Equivalently: the XOR (symmetric difference) of the prime sets of 
    # sf(210), sf(x_1), ..., sf(x_k) must be empty.
    # 
    # So if we represent each squarefree kernel as a bitmask over primes,
    # we need: mask(210) XOR mask(x_1) XOR ... XOR mask(x_k) = 0.
    # Which means: mask(x_1) XOR ... XOR mask(x_k) = mask(210).
    
    # For efficiency, let's work with the actual squarefree kernel values.
    # Two squarefree kernels "XOR to target" means their product divided by 
    # the square of their gcd equals the target.
    # Actually, for squarefree numbers a and b, a * b is a perfect square times 
    # sf(a) * sf(b) / gcd(sf(a), sf(b))^2... this is getting complicated.
    
    # Let me just use the direct approach: 
    # For a single number x: 210 * x is a perfect square iff sf(210 * x) == 1.
    # Since 210 and x are both integers, sf(210 * x) = 1 iff 210 * x is a perfect square.
    # 210 * x = k^2 => x = k^2 / 210. For x to be an integer, 210 must divide k^2.
    # Since 210 = 2 * 3 * 5 * 7, we need k to be divisible by 2*3*5*7 = 210.
    # So k = 210 * m for some m, and x = (210*m)^2 / 210 = 210 * m^2.
    # So x must be of the form 210 * m^2 for some integer m >= 1.
    # x > 210 means m >= 1 (since 210*1^2 = 210, but we need x > 210, so m >= 2).
    # Wait, x must be strictly larger than 210, so m >= 2 gives x = 210*4 = 840.
    # Let me verify: 210 * 840 = 176400 = 420^2. Yes!
    
    # So single-element solutions are x = 210 * m^2 for m >= 2.
    # Smallest is m=2: x = 840. max - 210 = 840 - 210 = 630.
    
    # For two elements a, b: 210 * a * b is a perfect square.
    # 210 * a * b = k^2.
    # This is more flexible. Let's think about it differently.
    # 
    # Let a = 210 * m^2 * u and b = 210 * n^2 * v where u and v are squarefree and 
    # sf(u) = sf(v) (so that u*v is a perfect square).
    # Actually, this is getting complicated. Let me just search.
    
    # Better approach: 
    # For each M from 211 upward, check if there exists a subset of {211, ..., M} 
    # of size <= 100 such that 210 * prod(subset) is a perfect square.
    # 
    # For size 1: check if any x in (210, M] has 210*x a perfect square.
    # For size 2: check if any pair (a,b) in (210, M]^2 has 210*a*b a perfect square.
    # etc.
    
    # Let's precompute for each x in (210, 10210]:
    #   sf[x] = squarefree kernel of x
    #   And we need: sf(210) * sf(x_1) * ... * sf(x_k) to be a perfect square.
    
    # Represent squarefree kernels as frozensets of primes, or as integers with 
    # prime factorization. Since squarefree kernels are products of distinct primes,
    # we can represent them as integers and check if their product is a perfect square.
    
    # For two numbers: 210 * a * b is a perfect square iff 
    # sf(210) * sf(a) * sf(b) is a perfect square.
    # Since all three are squarefree, their product is a perfect square iff 
    # each prime appears an even number of times.
    # 
    # Let's define: for squarefree integers s, t, u, the product s*t*u is a perfect 
    # square iff s*t*u = (s*t*u) / (gcd-related stuff)... 
    # Actually, simpler: s*t*u is a perfect square iff s*t*u = w^2 for some w.
    # Since s, t, u are squarefree, this means the multiset union of primes in s, t, u 
    # has each prime appearing 0 or 2 times.
    
    # Let me just compute directly:
    # For a pair (a, b): check if 210 * a * b is a perfect square.
    # This is O(1) per pair using math.isqrt.
    
    # Plan:
    # 1. For M from 211 to 10210:
    #    a. Check single: for x in (210, M], is 210*x a perfect square?
    #    b. Check pairs: for a < b in (210, M], is 210*a*b a perfect square?
    #    c. If found, return the solution.
    
    # But checking all pairs for large M is O(M^2) which is too slow for M up to 10210.
    
    # Smarter approach for pairs:
    # 210 * a * b is a perfect square iff b = k^2 / (210 * a) for some integer k,
    # and b must be an integer in (210, M].
    # So for each a, b must be of the form k^2 / (210*a).
    # Since b must be an integer, 210*a must divide k^2.
    # Let g = gcd(210*a, k). This is getting complicated.
    
    # Alternative: for each a in (210, M], compute what b values would work.
    # 210 * a * b = perfect square.
    # Let 210*a = s * t^2 where s is squarefree. Then we need s * b to be a perfect square,
    # which means b = s * u^2 for some u (since s is squarefree, s*b is a square iff 
    # b = s * u^2).
    # Wait, that's not quite right. If 210*a = s * t^2 with s squarefree, then 
    # 210*a*b = s * t^2 * b. For this to be a perfect square, s*b must be a perfect square.
    # Since s is squarefree, s*b is a perfect square iff b = s * v^2 for some integer v.
    # So for each a, the valid b values are b = s * v^2 where s = sf(210*a).
    
    # Let me implement this:
    
    best_result = None
    best_max = max_val + 1
    
    for M in range(n + 1, max_val + 1):
        if time.time() > deadline:
            break
        
        # Check single element
        found = False
        for x in range(n + 1, M + 1):
            prod = n * x
            r = math.isqrt(prod)
            if r * r == prod:
                if x > n and x <= M:
                    best_max = x
                    best_result = [x]
                    found = True
                    break
        if found:
            break
        
        # Check pairs
        # For each a in (210, M], find if there's a b in (210, M], b != a, such that 210*a*b is a perfect square.
        # Using the formula: if 210*a = s * t^2 (s squarefree), then b = s * v^2.
        
        # Precompute for current M: for each a, what are the candidate b values?
        # This is still potentially expensive. Let me try a different approach.
        
        # For each a in (210, M]:
        #   Compute s = sf(210 * a)
        #   Then b must be of the form s * v^2 for some v >= 1
        #   Check if any such b is in (210, M] and b != a
        
        for a in range(n + 1, M + 1):
            if time.time() > deadline:
                break
            
            # Compute sf(210 * a)
            prod_a = n * a
            s = _sf_kernel(prod_a)
            
            # b = s * v^2 for v >= 1
            # b must be in (210, M]
            # v^2 < M/s, so v < sqrt(M/s)
            max_v = math.isqrt(M // s) if s <= M else 0
            for v in range(1, max_v + 1):
                b = s * v * v
                if b > n and b <= M and b != a:
                    # Verify
                    prod = n * a * b
                    r = math.isqrt(prod)
                    if r * r == prod:
                        best_max = max(a, b)
                        best_result = [a, b]
                        found = True
                        break
            if found:
                break
        if found:
            break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Fallback
    return [840]
# EVOLVE-BLOCK-END
