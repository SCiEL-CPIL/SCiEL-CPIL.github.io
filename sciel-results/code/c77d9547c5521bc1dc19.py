import math

def construct():
    # Primes 1 mod 4
    primes = []
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0: return False
        return True
        
    # Generate primes 1 mod 4 up to 1000
    for i in range(2, 1000):
        if is_prime(i) and i % 4 == 1:
            primes.append(i)
            
    # DFS to find max product of (e_i+1)
    # State: index, remaining_log, current_product
    # We want to maximize current_product
    best_product = 1
    best_exponents = [0] * len(primes)
    
    # Use logs to avoid overflow
    MAX_LOG = 18 * math.log(10)
    
    def dfs(idx, rem_log, current_prod, exponents):
        nonlocal best_product, best_exponents
        if idx == len(primes):
            if current_prod > best_product:
                best_product = current_prod
                best_exponents = exponents[:]
            return
        
        p = primes[idx]
        log_p = math.log(p)
        
        # Try exponents 0 to max possible
        e = 0
        while e * log_p <= rem_log:
            # Pruning: even if we take max possible for this and all subsequent, can we beat best?
            # Simple pruning: if current_prod * (max_possible_for_rest) <= best_product, return
            # But we don't know max_possible_for_rest easily.
            
            dfs(idx + 1, rem_log - e * log_p, current_prod * (e + 1), exponents + [e])
            e += 1
            
    # Limit depth to first 20 primes to keep DFS fast
    dfs(0, MAX_LOG, 1, [])
    
    # Reconstruct N
    N = 1
    for i in range(len(primes)):
        if best_exponents[i] > 0:
            N *= primes[i] ** best_exponents[i]
            
    # N might be slightly less than 10^18, which is fine.
    # We need to find all pairs (x, y) such that x^2 + y^2 = N.
    # This is done by iterating x from 0 to sqrt(N/2) and checking if N-x^2 is a square.
    # But N is up to 10^18. sqrt(N) is 10^9.
    # Iterating 10^9 times is too slow in Python.
    
    # We need a faster way to find representations of N as sum of two squares.
    # Since we know the factorization of N, we can use the Cornacchia's algorithm or Gaussian integers.
    # But we need ALL representations.
    # The number of representations is 4 * best_product.
    # If best_product is ~1000, we have ~4000 representations.
    # We can generate them using the properties of Gaussian integers.
    
    # Simpler: Since we constructed N from primes 1 mod 4, we can use the fact that each prime has a unique representation.
    # But implementing the full generation is complex.
    
    # Alternative: If best_product is small enough, we can just iterate?
    # If N is small enough? No, N is ~10^18.
    
    # Let's check the size of best_product.
    # If we use 11 primes with e=1, product is 2^11 = 2048.
    # Iterating 10^9 is too slow.
    
    # We need to generate the pairs (x,y) directly from the factorization.
    # This is a standard problem.
    
    # For now, let's just return the square pairs if we can find them efficiently.
    # If we can't, we fall back to the original method with a larger limit?
    
    # Actually, we can use a randomized search or a smarter iteration?
    # No, we need all pairs.
    
    # Let's try to increase the limit of the original method to see if we can get more pairs without complex math.
    # Original method score 5.8 with LIMIT=5e7.
    # If we use LIMIT=1e9, M~30000, M^2=9e8. Too slow.
    
    # The Gaussian integer approach is the only way to get >100 pairs quickly.
    
    # Implementation of finding all representations of N as sum of two squares given factorization.
    # This is complex.
    
    # Let's try a different tactic:
    # The problem allows any N.
    # What if we pick a smaller N with many representations?
    # e.g. N = 5^5 * 13^3 ...
    # If N is smaller, sqrt(N) is smaller, and we can brute force x.
    # Max N for brute force in 120s?
    # Python loop ~10^7/sec. 120s -> 10^9 iterations.
    # So we can brute force up to N = 10^18 if we are lucky? No, sqrt(10^18)=10^9.
    # So we can brute force up to N=10^18 if we only iterate sqrt(N).
    # 10^9 iterations in Python is ~100-200 seconds. Borderline.
    
    # So we can pick N <= 10^18 and brute force x from 0 to isqrt(N).
    # We just need to find an N with many representations.
    # The N constructed above (product of primes 1 mod 4) has ~1000 representations.
    # We can find them by iterating x?
    # If N=10^18, isqrt(N)=10^9.
    # Looping 10^9 times in Python is too slow.
    
    # We need to reduce the search space.
    # We know N is a sum of two squares.
    # We can use the property that x and y must be divisors of something? No.
    
    # Let's stick to the original approach but with a much larger LIMIT and optimized C code?
    # We only have Python.
    
    # What if we use the fact that we only need to return the pairs?
    # If we find N with 1000 pairs, we need to list them.
    
    # Let's try to find a smaller N with many pairs.
    # Max pairs for N < 10^6?
    # If N is small, we can brute force.
    # But max pairs is small.
    
    # The key is to find N such that sqrt(N) is small enough to brute force, but N has many representations.
    # This is a contradiction. Many representations imply large N (usually).
    
    # However, we can use the Gaussian integer generation if we implement it.
    # It's worth the effort.
    
    # But for this edit, let's just increase the LIMIT and hope for a better N, or use a smarter filter.
    
    # Actually, the current code is too slow for large LIMIT.
    # Let's optimize the current code to run faster.
    # Use `array` or `numpy`?
    
    # Let's try to find N with many pairs by checking specific candidates.
    # Candidates: N = 2^k, N = 3^k, N = 5^k...
    # N = 5^k has few representations.
    
    # I will implement the Gaussian integer method to find pairs for the constructed N.
    
    pass
