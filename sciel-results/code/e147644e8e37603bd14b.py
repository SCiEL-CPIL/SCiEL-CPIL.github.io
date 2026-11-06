import time
from math import isqrt

def construct():
    start_time = time.time()
    
    # Primes p = 1 mod 4
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101]
    
    # Function to find x, y such that x^2 + y^2 = p
    def sum_of_two_squares_prime(p):
        for x in range(1, p):
            if (x * x) % p == p - 1:
                for a in range(1, isqrt(p) + 1):
                    b2 = p - a * a
                    b = isqrt(b2)
                    if b * b == b2:
                        return a, b
        return 1, 0

    prime_reps = []
    for p in primes:
        x, y = sum_of_two_squares_prime(p)
        prime_reps.append((x, y))
        
    reps = [(1, 0)]
    
    for x, y in prime_reps:
        new_reps = []
        for u, v in reps:
            u1 = u * x - v * y
            v1 = u * y + v * x
            new_reps.append((u1, v1))
            
            u2 = u * x + v * y
            v2 = v * x - u * y
            new_reps.append((u2, v2))
        reps = new_reps
        
    pairs_set = set()
    for u, v in reps:
        a = 4 * u * u
        b = 4 * v * v
        if a > b:
            a, b = b, a
        pairs_set.add((a, b))
        
    pairs = sorted(pairs_set)
    
    M = 1
    for p in primes:
        M *= p
    N = 4 * M
    
    return {'N': N, 'pairs': [[a, b] for a, b in pairs]}
