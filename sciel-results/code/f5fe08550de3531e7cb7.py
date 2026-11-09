import time
import math

def construct():
    start_time = time.time()
    deadline = 55  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    
    def largest_prime_factor(n):
        """Compute the largest prime factor of n using trial division."""
        if n <= 1:
            return 1
        lpf = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                lpf = d
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            lpf = n
        return lpf
    
    def generate_smooth(prime_list, min_val, max_val):
        """Generate all numbers in [min_val, max_val] whose prime factors are in prime_list."""
        smooths = set()
        def gen(idx, current):
            if current > max_val:
                return
            if current >= min_val:
                smooths.add(current)
            if idx >= len(prime_list):
                return
            p = prime_list[idx]
            val = current
            while val <= max_val:
                gen(idx + 1, val)
                val *= p
                if val > max_val:
                    break
        gen(0, 1)
        return smooths
    
    # Try increasing smoothness thresholds
    prime_thresholds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for p in prime_thresholds:
        current_primes = [x for x in prime_thresholds if x <= p]
        smooths = generate_smooth(current_primes, N_MIN, N_MAX + 1)
        
        if not smooths:
            continue
            
        # Check for consecutive pairs (n, n+1) where both are p-smooth
        for x in smooths:
            if x + 1 in smooths and N_MIN <= x <= N_MAX:
                return x
        
        if time.time() - start_time > deadline:
            break
    
    # Fallback: brute-force search with early termination
    # Use a sieve to precompute smallest prime factors for efficiency
    if time.time() - start_time < deadline - 5:
        # Precompute largest prime factor up to N_MAX + 1
        max_val = N_MAX + 1
        lpf_arr = [1] * (max_val + 1)
        
        for i in range(2, max_val + 1):
            if lpf_arr[i] == 1:
                for j in range(i, max_val + 1, i):
                    lpf_arr[j] = i
        
        best_n = N_MIN
        best_lpf = float('inf')
        
        for n in range(N_MIN, N_MAX + 1):
            lpf_n = lpf_arr[n]
            lpf_np1 = lpf_arr[n + 1]
            combined_lpf = max(lpf_n, lpf_np1)
            
            if combined_lpf < best_lpf:
                best_lpf = combined_lpf
                best_n = n
            
            if time.time() - start_time > deadline:
                break
        
        return best_n
    
    return N_MIN
