"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n."""
    if n == 1:
        return 1
    result = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            result *= (d ** (exp + 1) - 1) // (d - 1)
        d += 1
    if temp > 1:
        result *= (temp * temp - 1) // (temp - 1)
    return result

def _sigma_fast(n, primes):
    """Compute sigma(n) using a list of primes."""
    if n == 1:
        return 1
    result = 1
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            result *= (p ** (exp + 1) - 1) // (p - 1)
    if temp > 1:
        result *= (temp * temp - 1) // (temp - 1)
    return result

def _generate_smooth_numbers(max_val, max_prime_exp=6):
    """Generate smooth numbers up to max_val using small primes."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    numbers = set()
    
    def gen(idx, current):
        if current > max_val:
            return
        if current > 1:
            numbers.add(current)
        if idx >= len(primes):
            return
        p = primes[idx]
        val = current
        for e in range(max_prime_exp + 1):
            if val > max_val:
                break
            gen(idx + 1, val)
            val *= p
    
    gen(0, 1)
    return numbers

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s
    
    # Generate primes up to a reasonable limit for factorization
    sieve_limit = 100000
    is_prime = [True] * (sieve_limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(sieve_limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, sieve_limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, sieve_limit + 1) if is_prime[i]]
    
    # Generate candidate k values
    # Strategy: generate smooth numbers and also numbers in ranges
    candidates = set()
    
    # Smooth numbers up to ~10^7
    smooth_max = 10_000_000
    smooth = _generate_smooth_numbers(smooth_max, max_prime_exp=8)
    candidates.update(smooth)
    
    # Add some random-ish structured numbers
    for base in range(1, 200):
        for mult in range(1, 100):
            val = base * mult
            if val <= 10_000_000:
                candidates.add(val)
    
    # Add numbers that are products of small primes with various exponents
    small_primes = [2, 3, 5, 7, 11, 13]
    def gen_combo(idx, current):
        if current > 10_000_000:
            return
        if current > 1:
            candidates.add(current)
        if idx >= len(small_primes):
            return
        p = small_primes[idx]
        val = current
        for e in range(10):
            if val > 10_000_000:
                break
            gen_combo(idx + 1, val)
            val *= p
    gen_combo(0, 1)
    
    # Compute M = k * sigma(k) for all candidates
    m_map = defaultdict(list)
    for k in candidates:
        if k <= 0:
            continue
        s = _sigma_fast(k, primes)
        M = k * s
        if M <= 10**18:
            m_map[M].append(k)
    
    # Find M with most solutions
    best_M = 333312
    best_ks = [336, 372, 434]
    
    for M, ks in m_map.items():
        if len(ks) > len(best_ks):
            # Verify all
            valid_ks = []
            for k in ks:
                if k <= math.isqrt(M) and k * _sigma_fast(k, primes) == M:
                    valid_ks.append(k)
            if len(valid_ks) > len(best_ks):
                best_M = M
                best_ks = valid_ks
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Also try to find more solutions for the best M
    if best_M > 0 and time.time() < deadline - 20:
        # Try to find more k for best_M
        sqrt_M = math.isqrt(best_M)
        # Check divisors of best_M
        n = best_M
        divisors = []
        d = 1
        while d * d <= n:
            if n % d == 0:
                divisors.append(d)
                if d != n // d:
                    divisors.append(n // d)
            d += 1
        
        all_valid = []
        for k in divisors:
            if k <= sqrt_M:
                s = _sigma_fast(k, primes)
                if k * s == best_M:
                    all_valid.append(k)
        
        if len(all_valid) > len(best_ks):
            best_ks = all_valid
    
    # Ensure types are correct
    best_M = int(best_M)
    best_ks = list(set(int(k) for k in best_ks))
    best_ks.sort()
    
    # Final verification
    final_ks = []
    for k in best_ks:
        if 1 <= k <= math.isqrt(best_M):
            s = _sigma_fast(k, primes)
            if k * s == best_M:
                final_ks.append(k)
    
    if len(final_ks) == 0:
        # Fallback to seed
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_M, 'k': final_ks}
# EVOLVE-BLOCK-END
