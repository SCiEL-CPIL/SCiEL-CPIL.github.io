import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 210
    max_val = 10210

    # 210 = 2 * 3 * 5 * 7, so its square-free part is 210 itself
    # We need a set S of distinct integers in [211, 10210] such that
    # 210 * prod(S) is a perfect square.
    # This means prod(S) must have the same square-free part as 210 (i.e., 210)
    # OR more precisely, the square-free part of prod(S) must equal 210.
    
    # Actually, let's think in terms of prime parity vectors.
    # 210 = 2^1 * 3^1 * 5^1 * 7^1
    # For 210 * prod(S) to be a square, for each prime p, the total exponent must be even.
    # So for p in {2,3,5,7}: (1 + sum of exponents in S) must be even => sum of exponents in S must be odd.
    # For p not in {2,3,5,7}: sum of exponents in S must be even.

    # Let's find the smallest possible max value.
    # Try single element: x such that 210*x is a square.
    # 210*x = 2*3*5*7*x must be a square.
    # So x must be 2*3*5*7*k^2 = 210*k^2 for some k.
    # Smallest x > 210: k=1 gives x=210 (not > 210), k=2 gives x=840.
    # So [840] gives max-210 = 630.

    # Can we do better with 2 elements? x*y such that 210*x*y is a square.
    # We want to minimize max(x,y).
    # 210*x*y = square means x*y has square-free part 210.
    # So x = 210*a^2 * b, y = 210*c^2 * d where b*d has square-free part 1 (i.e., b*d is a square).
    # More simply: let x = s1 * u^2, y = s2 * v^2 where s1, s2 are square-free.
    # Then 210 * s1 * s2 must be a square, so s1 * s2 = 210 * (square).
    # Since s1, s2 are square-free, s1 * s2 = 210.
    # So s1 and s2 are complementary square-free factors of 210.
    # Pairs (s1, s2): (1,210), (2,105), (3,70), (5,42), (6,35), (7,30), (10,21), (14,15)
    # For each pair, x = s1*u^2 > 210, y = s2*v^2 > 210.
    # We want to minimize max(x,y).
    
    best_answer = [840]  # max-210 = 630
    best_raw = 630

    def try_pair(s1, s2):
        """Find x = s1*u^2 > 210, y = s2*v^2 > 210 minimizing max(x,y)."""
        nonlocal best_answer, best_raw
        # x = s1 * u^2 > 210 => u^2 > 210/s1 => u > sqrt(210/s1)
        min_u = math.isqrt(math.ceil(210 / s1))
        if min_u * min_u <= 210 / s1:
            min_u += 1
        min_u = max(min_u, 1)
        
        min_v = math.isqrt(math.ceil(210 / s2))
        if min_v * min_v <= 210 / s2:
            min_v += 1
        min_v = max(min_v, 1)
        
        # Try small u, v
        for u in range(min_u, min_u + 20):
            x = s1 * u * u
            if x > max_val:
                break
            for v in range(min_v, min_v + 20):
                y = s2 * v * v
                if y > max_val:
                    break
                if x == y:
                    continue
                m = max(x, y)
                if m - 210 < best_raw:
                    # Verify
                    prod = 210 * x * y
                    r = math.isqrt(prod)
                    if r * r == prod:
                        best_raw = m - 210
                        best_answer = [x, y]
                        return

    # Try all complementary pairs
    pairs = [(1, 210), (2, 105), (3, 70), (5, 42), (6, 35), (7, 30), (10, 21), (14, 15)]
    for s1, s2 in pairs:
        try_pair(s1, s2)

    # Try 3 elements? Might be able to get lower max.
    # With 3 elements x,y,z: 210*x*y*z is a square.
    # This is more complex. Let's first see if we can beat 630 with 2 elements.
    
    # Let's also try: can we find a single element smaller than 840?
    # x = 210*k^2, smallest k>1 is k=2, x=840. No smaller single element.

    # For 2 elements, let's be more thorough.
    # Actually, let me reconsider. The square-free part of x*y must be 210.
    # x*y = 210 * k^2 for some integer k.
    # So x*y = 210*k^2, with x,y > 210, x != y, x,y <= 10210.
    # We want to minimize max(x,y).
    # Let's iterate k and factor 210*k^2.
    
    for k in range(1, 100):
        if time.time() > deadline:
            break
        val = 210 * k * k
        if val > max_val * max_val:
            break
        # Find factor pairs (x,y) of val with x <= y, x > 210, y > 210, x != y
        # x * y = val, x <= sqrt(val)
        sqrt_val = math.isqrt(val)
        for x in range(max(211, val // max_val), min(sqrt_val, max_val) + 1):
            if time.time() > deadline:
                break
            if val % x == 0:
                y = val // x
                if y > 210 and y <= max_val and x != y:
                    m = max(x, y)
                    if m - 210 < best_raw:
                        prod = 210 * x * y
                        r = math.isqrt(prod)
                        if r * r == prod:
                            best_raw = m - 210
                            best_answer = [x, y]

    # Try 3 elements more systematically
    # 210 * x * y * z = square
    # Let's use the parity vector approach for small numbers.
    # But first, let's check if we can get max < 840 with 2 elements.
    # The above loop should have found it if it exists.

    # Let's try a more targeted approach for 3 elements.
    # We need x*y*z to have square-free part 210.
    # Let's try to find x,y,z in [211, M] for small M.
    
    # Precompute square-free parts
    def square_free_part(x):
        sf = 1
        temp = x
        p = 2
        while p * p <= temp:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                sf *= p
            p += 1
        if temp > 1:
            sf *= temp
        return sf

    # For 3 elements, we need sf(x) * sf(y) * sf(z) to have square-free part 210.
    # This means sf(x) * sf(y) * sf(z) = 210 * k^2 for some k.
    # Equivalently, the product of the square-free parts, when reduced, gives 210.
    
    # Let's try: for M from 211 upward, check if there exist 3 distinct numbers in [211, M]
    # whose product with 210 is a square.
    
    # Use a map: for each possible "partial product square-free part", track which pairs achieve it.
    # sf(x) * sf(y) mod squares = some value, then we need sf(z) to complement to 210.
    
    # Actually, let's use the XOR of prime exponent parities.
    # For each number, compute its parity vector over all primes up to 10210.
    # But that's a huge vector. Instead, use the square-free part as a proxy.
    # sf(a) * sf(b) has square-free part = sf(a*b) = sf(a)*sf(b) / (greatest square dividing sf(a)*sf(b))
    # This is getting complex. Let's just brute force for small M with 3 elements.
    
    if time.time() < deadline - 5:
        # Try 3 elements with max up to ~500
        # Use a dictionary: key = square-free part of product of 2 elements, value = (x, y)
        # For each z, check if sf(x*y) * sf(z) has square-free part 210.
        
        # Precompute sf for range
        sf_cache = {}
        for i in range(211, 600):
            sf_cache[i] = square_free_part(i)
        
        # For each pair (x,y) with x < y, compute sf(x*y)
        # sf(x*y) = square_free_part(x*y)
        # But x*y can be large. Let's compute it directly.
        
        # Alternative: use the fact that sf(a)*sf(b) reduced gives sf(a*b).
        # Let's just compute sf(x*y) directly for x,y in [211, M].
        
        # For efficiency, for each z, we need sf(x*y) such that sf(x*y*z) = 210.
        # sf(x*y*z) = square_free_part(x*y*z)
        # This is hard to invert. Let's just try small M.
        
        # Try M up to 400 for 3 elements
        for M in range(300, 500):
            if time.time() > deadline - 2:
                break
            # Build all pairs
            pair_sf = {}
            for x in range(211, M):
                for y in range(x + 1, M + 1):
                    prod_xy = x * y
                    sf_xy = square_free_part(prod_xy)
                    if sf_xy not in pair_sf:
                        pair_sf[sf_xy] = (x, y)
            
            # For each z, check if there's a pair with sf_xy such that sf(xy*z) = 210
            for z in range(211, M + 1):
                if time.time() > deadline - 1:
                    break
                # We need sf(x*y*z) = 210
                # sf(x*y*z) = square_free_part(x*y*z)
                # This is expensive to check for all pairs.
                # Instead, note that sf(x*y*z) = 210 means x*y*z = 210 * k^2
                # So x*y = 210 * k^2 / z, which must be an integer.
                # This approach is also complex.
                pass
        
        # Let's try a different approach: for 3 elements, just check specific candidates.
        # Actually, let's see if 2 elements can beat 630.
        # From the k-loop above, we should have found the best 2-element solution.
    
    # Let's verify our best answer
    if best_answer:
        prod = 210
        for x in best_answer:
            prod *= x
        r = math.isqrt(prod)
        if r * r == prod and 1 <= len(best_answer) <= 100:
            if all(isinstance(x, int) and 210 < x <= 10210 for x in best_answer):
                return [int(x) for x in best_answer]
    
    return [840]
