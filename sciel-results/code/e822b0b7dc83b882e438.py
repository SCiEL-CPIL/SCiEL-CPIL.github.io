def construct():
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    def small_prime_part(x):
        result = x
        for p in small_primes:
            while result % p == 0:
                result //= p
        return x // result
    
    best_n = 1000
    best_count = 25
    
    for n in range(1000, 1000001):
        factors = set()
        for i in range(1, 25):
            factors.add(small_prime_part(n + i))
        count = len(factors)
        if count < best_count:
            best_count = count
            best_n = n
            if best_count <= 2:  # unlikely to do much better
                # Could continue to see if 1 is possible, but let's break early for speed
                # Actually, let's keep searching but this is a good early exit heuristic
                pass
    
    return best_n
