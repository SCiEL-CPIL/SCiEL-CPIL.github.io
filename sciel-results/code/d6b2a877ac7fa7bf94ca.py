"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    MAX_VAL = N_MAX + 1  # We need LPF up to 1000001
    
    # Precompute primes up to sqrt(MAX_VAL) for trial division
    limit = int(math.isqrt(MAX_VAL)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    small_primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Function to compute largest prime factor using trial division by small primes
    def lpf_trial_div(n):
        if n <= 1:
            return 1
        original_n = n
        max_pf = 1
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                max_pf = p
                while n % p == 0:
                    n //= p
        if n > 1:
            max_pf = max(max_pf, n)
        return max_pf
    
    # Precompute LPF for all numbers using the trial division method
    # This is slower per number but avoids the full sieve memory overhead
    # and allows us to focus on finding smooth pairs
    
    lpf = [0] * (MAX_VAL + 1)
    lpf[0] = 0
    lpf[1] = 1
    
    # Use a more efficient approach: segmented sieve for LPF
    # Process in segments to reduce memory pressure and allow early termination
    SEG_SIZE = 100000
    best_n = N_MIN
    best_val = float('inf')
    
    for seg_start in range(0, MAX_VAL + 1, SEG_SIZE):
        seg_end = min(seg_start + SEG_SIZE, MAX_VAL + 1)
        
        # Check time
        if time.time() > deadline:
            break
        
        # For each segment, compute LPF for numbers in [seg_start, seg_end)
        seg_lpf = [0] * (seg_end - seg_start)
        
        # Initialize with the number itself (assume prime initially)
        for i in range(seg_start, seg_end):
            if i >= 2:
                seg_lpf[i - seg_start] = i
        
        # Sieve with small primes
        for p in small_primes:
            if p > seg_end:
                break
            # Find first multiple of p in this segment
            first_mult = ((seg_start + p - 1) // p) * p
            if first_mult < seg_start:
                first_mult = seg_start
            
            for j in range(first_mult, seg_end, p):
                if j >= 2:
                    seg_lpf[j - seg_start] = p  # p is a prime factor, and since we go in order, last one is largest
        
        # Now scan for best consecutive pair in this segment
        for n in range(max(N_MIN, seg_start), min(N_MAX, seg_end - 1)):
            if time.time() > deadline:
                return best_n
            
            # Get LPF for n and n+1
            lpf_n = seg_lpf[n - seg_start] if n - seg_start < len(seg_lpf) else lpf_trial_div(n)
            lpf_n1 = seg_lpf[n + 1 - seg_start] if (n + 1) - seg_start < len(seg_lpf) else lpf_trial_div(n + 1)
            
            current_val = max(lpf_n, lpf_n1)
            
            if current_val < best_val:
                best_val = current_val
                best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
