import math
from collections import defaultdict

def construct():
    # Find primes p == 1 mod 4
    primes_1mod4 = []
    n = 5
    while True:
        if n % 4 == 1:
            # Check if prime
            is_prime = True
            for i in range(3, int(math.isqrt(n)) + 1, 2):
                if n % i == 0:
                    is_prime = False
                    break
            if is_prime:
                # Check product limit
                # We will accumulate and break
                primes_1mod4.append(n)
        n += 2
        
        # Stop if we have enough or product is too big
        # We need to calculate product to decide
        # But we don't know which ones to keep yet.
        # Let's just generate a list of candidates and pick the best subset.
        # Since we want smallest primes, we just take the first k.
        if len(primes_1mod4) > 20:
            break
            
    # Select largest k such that product <= 10^18
    best_k = 0
    prod_val = 1
    for i, p in enumerate(primes_1mod4):
        if prod_val * p <= 10**18:
            prod_val *= p
            best_k = i + 1
        else:
            break
            
    N = prod_val
    selected_primes = primes_1mod4[:best_k]
    
    # Find representations for each prime
    # p = a^2 + b^2
    prime_reps = []
    for p in selected_primes:
        found = False
        for a in range(1, int(math.isqrt(p // 2)) + 1):
            b2 = p - a * a
            b = int(math.isqrt(b2))
            if b * b == b2:
                prime_reps.append((a, b))
                found = True
                break
        if not found:
            # Should not happen for p == 1 mod 4
            prime_reps.append((1, 0)) # Fallback, shouldn't be used

    # Combine representations
    # Start with (1, 0)
    reps = [(1, 0)]
    
    for a, b in prime_reps:
        new_reps = []
        for u, v in reps:
            # (u + iv)(a + ib) = (ua - vb) + i(ub + va)
            u1 = u * a - v * b
            v1 = u * b + v * a
            new_reps.append((u1, v1))
            
            # (u + iv)(a - ib) = (ua + vb) + i(va - ub)
            u2 = u * a + v * b
            v2 = v * a - u * b
            new_reps.append((u2, v2))
        reps = new_reps
        
    # Filter and create pairs
    pairs = []
    seen = set()
    for u, v in reps:
        # We need u^2 + v^2 = N
        # And u, v > 0
        # And u < v (to ensure a <= b and distinct)
        if u <= 0 or v <= 0:
            continue
        if u >= v:
            continue
        a_val = u * u
        b_val = v * v
        if a_val + b_val != N:
            continue # Safety check
            
        pair = (a_val, b_val)
        if pair not in seen:
            seen.add(pair)
            pairs.append([a_val, b_val])
            
    return {'N': N, 'pairs': pairs}
