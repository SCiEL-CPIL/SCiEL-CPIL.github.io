import time
import math
import heapq

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit
    
    L = 10000
    R = 1000000
    
    max_val = R + 1
    
    # Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Get list of primes up to max_val
    primes = [i for i in range(2, max_val + 1) if spf[i] == i]
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        max_p = 1
        while n > 1:
            p = spf[n]
            if p > max_p:
                max_p = p
            while n % p == 0:
                n //= p
        return max_p
        
    def generate_smooth_numbers(bound_prime, limit):
        # Generate all numbers <= limit with prime factors <= bound_prime
        # using a min-heap and a set to avoid duplicates
        heap = [1]
        seen = {1}
        smooth_nums = []
        
        # Primes <= bound_prime
        primes_subset = [p for p in primes if p <= bound_prime]
        
        while heap:
            x = heapq.heappop(heap)
            if x > limit:
                break
            if x >= L:
                smooth_nums.append(x)
                
            for p in primes_subset:
                y = x * p
                if y > limit:
                    break
                if y not in seen:
                    seen.add(y)
                    heapq.heappush(heap, y)
                    
        return sorted(smooth_nums)

    best_n = L
    best_val = float('inf')
    
    # Iterate over possible maximum prime factors (primes)
    for p in primes:
        if time.time() > deadline:
            break
            
        if p >= best_val:
            break
            
        smooth_nums = generate_smooth_numbers(p, R)
        
        # Check for consecutive pairs
        for i in range(len(smooth_nums) - 1):
            a = smooth_nums[i]
            b = smooth_nums[i + 1]
            if b == a + 1:
                # Valid pair (a, a+1)
                # Calculate exact largest prime factor
                val = max(largest_prime_factor(a), largest_prime_factor(b))
                if val < best_val:
                    best_val = val
                    best_n = a
                    
    return best_n
# EVOLVE-BLOCK-END
