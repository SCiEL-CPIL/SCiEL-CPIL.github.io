import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    max_m = 10**18
    
    # Sieve for sigma(k)
    L = 2 * 10**6
    
    sigma = [0] * (L + 1)
    
    # Compute sum of divisors for all numbers up to L
    for d in range(1, L + 1):
        for m in range(d, L + 1, d):
            sigma[m] += d
            
    # Map M = k * sigma(k) to list of k
    solutions = defaultdict(list)
    
    for k in range(1, L + 1):
        m_val = k * sigma[k]
        if m_val <= max_m:
            solutions[m_val].append(k)
            
    # Sort candidates by number of solutions, take top few
    candidates = sorted(solutions.items(), key=lambda x: -len(x[1]))
    
    best_m = 0
    best_k_list = []
    
    # Precompute small primes for fast factorization
    small_primes = []
    is_prime_arr = [True] * (L + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, L + 1):
        if is_prime_arr[i]:
            small_primes.append(i)
            if i * i <= L:
                for j in range(i * i, L + 1, i):
                    is_prime_arr[j] = False
    
    # For each top candidate, try to find additional k values
    for m_val, k_list in candidates[:100]:
        if time.time() > deadline - 10:
            break
            
        # Try to find more k values by checking divisors of m_val
        sqrt_m = math.isqrt(m_val)
        
        if len(k_list) >= 2 and time.time() < deadline - 30:
            # Factor m_val to find divisors
            divisors = _factor_to_divisors(m_val, small_primes, deadline)
            if divisors is None:
                continue
                
            # Check each divisor k of m_val where k <= sqrt_m
            existing = set(k_list)
            for k in divisors:
                if time.time() > deadline - 5:
                    break
                if k in existing or k > sqrt_m or k < 1:
                    continue
                if k <= L:
                    sk = sigma[k]
                else:
                    sk = _sigma_fast(k, small_primes)
                if k * sk == m_val:
                    existing.add(k)
                    k_list.append(k)
        
        if len(k_list) > len(best_k_list):
            best_m = m_val
            best_k_list = list(k_list)
    
    # Fallback if no solutions found
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Ensure k_i <= sqrt(M)
    sqrt_m = math.isqrt(best_m)
    valid_k = sorted(set(k for k in best_k_list if k <= sqrt_m and k >= 1))
    
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

def _is_prime_mr(n):
    """Miller-Rabin primality test for n < 2^64."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def _factor_to_divisors(n, small_primes, deadline):
    """Factor n and return list of all divisors. Returns None if too slow."""
    divisors = [1]
    remaining = n
    for p in small_primes:
        if p * p > remaining:
            break
        if time.time() > deadline - 5:
            return None
        if remaining % p == 0:
            exp = 0
            while remaining % p == 0:
                remaining //= p
                exp += 1
            new_divs = []
            pow_p = 1
            for i in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * pow_p)
                pow_p *= p
            divisors = new_divs
    if remaining > 1:
        if _is_prime_mr(remaining):
            new_divs = []
            for d in divisors:
                new_divs.append(d)
                new_divs.append(d * remaining)
            divisors = new_divs
        else:
            p = small_primes[-1] + 2 if small_primes else 2
            if p % 2 == 0:
                p += 1
            found = False
            while p * p <= remaining:
                if time.time() > deadline - 5:
                    return None
                if remaining % p == 0:
                    exp = 0
                    while remaining % p == 0:
                        remaining //= p
                        exp += 1
                    new_divs = []
                    pow_p = 1
                    for i in range(exp + 1):
                        for d in divisors:
                            new_divs.append(d * pow_p)
                        pow_p *= p
                    divisors = new_divs
                    found = True
                    break
                p += 2
            if not found and remaining > 1:
                new_divs = []
                for d in divisors:
                    new_divs.append(d)
                    new_divs.append(d * remaining)
                divisors = new_divs
    return divisors

def _sigma_fast(n, small_primes):
    """Compute sigma(n) using precomputed small primes."""
    if n <= 0:
        return 0
    result = 1
    for p in small_primes:
        if p * p > n:
            break
        if n % p == 0:
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            result *= power_sum
    if n > 1:
        result *= (1 + n)
    return result

if __name__ == "__main__":
    result = construct()
    print(result)
