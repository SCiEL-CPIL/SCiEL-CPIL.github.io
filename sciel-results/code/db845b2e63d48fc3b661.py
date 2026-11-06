import time
import math
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 100

    # Primes p = 1 mod 4
    # We need k such that 2^k gives enough pairs.
    # k=11 gives 2048 representations in Z[i], which maps to ~1024 pairs {x^2, y^2}.
    primes_1_4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97]
    
    # Check N size
    N = 1
    for p in primes_1_4:
        N *= p
        
    if N > 10**18:
        # If too large, reduce k
        primes_1_4 = primes_1_4[:10]
        N = 1
        for p in primes_1_4:
            N *= p
            
    # Find a, b for each p = a^2 + b^2
    def sum_of_two_squares(p):
        # p is prime, p = 1 mod 4
        # Find x such that x^2 = -1 mod p
        # Then use Euclidean algorithm
        # Or just brute force for small p
        for a in range(1, int(math.isqrt(p)) + 1):
            b2 = p - a * a
            b = int(math.isqrt(b2))
            if b * b == b2:
                return a, b
        return None, None

    factors = []
    for p in primes_1_4:
        a, b = sum_of_two_squares(p)
        if a is None:
            continue
        # pi = a + i*b
        factors.append((a, b))
        
    # Generate all 2^k combinations
    # z = prod (a_i + i*b_i or a_i - i*b_i)
    # We can represent z as (u, v)
    
    pairs = set()
    
    # Start with z = 1 + 0i
    # Iterate through all 2^k sign choices for the imaginary part of each factor?
    # Actually, choosing pi or bar(pi) is equivalent to choosing the sign of the imaginary part
    # if we fix the real part? No.
    # pi = a + ib. bar(pi) = a - ib.
    # So we choose s_i in {1, -1} for the imaginary part of the i-th factor.
    # The real part is always a_i? No.
    # Wait, the factors are complex numbers.
    # The set of representations is { prod (a_i + i s_i b_i) | s_i in {1, -1} }.
    # Is this correct?
    # N = prod (a_i^2 + b_i^2) = prod (a_i + i b_i)(a_i - i b_i).
    # Any factor z of N in Z[i] with norm N is a product of one from each pair {pi_i, bar(pi_i)}.
    # So yes, z = prod_{i} (a_i + i s_i b_i) where s_i is +1 or -1.
    # Note: This generates 2^k values.
    # Due to symmetry, z and -z, i*z, -i*z give same x^2, y^2.
    # Also z and bar(z) give (x, y) and (x, -y) -> same x^2, y^2.
    # So we just collect {x^2, y^2}.
    
    k = len(factors)
    
    # Use recursion or product
    # To avoid memory issues with large k, we can do it iteratively or use product.
    # k=11, 2^11 = 2048. product is fine.
    
    for signs in product([1, -1], repeat=k):
        u = 1
        v = 0
        for i in range(k):
            a, b = factors[i]
            s = signs[i]
            # (u + iv) * (a + i s b)
            # real: u*a - v*(s*b)
            # imag: u*(s*b) + v*a
            new_u = u * a - v * s * b
            new_v = u * s * b + v * a
            u, v = new_u, new_v
            
        x2 = u * u
        y2 = v * v
        # Ensure a <= b for the pair [a, b] in the output
        # The task requires pairs [a, b] with a <= b
        if x2 > y2:
            x2, y2 = y2, x2
        pairs.add((x2, y2))
        
    # Convert to list of lists
    pair_list = [list(p) for p in pairs]
    
    # Verify N <= 10^18
    if N > 10**18:
        # Fallback to a smaller valid N if the construction failed size check
        # But we checked above.
        pass
        
    # Ensure at least one pair
    if not pair_list:
        # Fallback: N=2, pair [1,1]
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Optional: Check time
    if time.time() > deadline:
         # Return what we have
         pass

    return {'N': N, 'pairs': pair_list}
