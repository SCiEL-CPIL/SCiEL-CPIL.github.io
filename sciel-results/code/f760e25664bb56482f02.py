import time
import math
import sys
from array import array
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    max_m = 10**18
    
    # Phase 1: Sieve-based search
    # Increase L to find more base collisions
    L = 4 * 10**6
    
    # Sieve for sigma(k)
    # Using array('q') for memory efficiency and speed
    sigma = array('q', [0]) * (L + 1)
    
    # Optimized sieve loop
    for d in range(1, L + 1):
        for m in range(d, L + 1, d):
            sigma[m] += d
            
    # Map M = k * sigma(k) to list of k
    solutions = defaultdict(list)
    
    for k in range(1, L + 1):
        m_val = k * sigma[k]
        if m_val <= max_m:
            solutions[m_val].append(k)
            
    # Find top candidates by number of solutions
    candidates = []
    for m, k_list in solutions.items():
        if len(k_list) >= 2:
            candidates.append((len(k_list), m, k_list))
    
    # Sort by count descending, then by M ascending
    candidates.sort(key=lambda x: (-x[0], x[1]))
    
    # Helper functions for exact verification
    
    def factorize(n):
        """Return prime factorization of n as dict {prime: exponent}"""
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def sigma_from_factors(factors):
        """Compute sigma from prime factorization"""
        result = 1
        for p, e in factors.items():
            result *= (p**(e+1) - 1) // (p - 1)
        return result

    def get_divisors(n):
        """Generate all divisors of n"""
        if n <= 0:
            return []
        factors = factorize(n)
        primes = list(factors.keys())
        exponents = [factors[p] for p in primes]
        
        divisors = [1]
        for i, p in enumerate(primes):
            current_divs = list(divisors)
            power = 1
            for e in range(1, exponents[i] + 1):
                power *= p
                for d in current_divs:
                    divisors.append(d * power)
        return divisors

    def find_all_solutions_for_m(m):
        """Find ALL k such that k*sigma(k) = m and k <= sqrt(m)"""
        sqrt_m = math.isqrt(m)
        if sqrt_m < 1:
            return []
        
        divs = get_divisors(m)
        
        results = []
        for k in divs:
            if k > sqrt_m:
                continue
            fac = factorize(k)
            sig = sigma_from_factors(fac)
            if k * sig == m:
                results.append(k)
        
        return sorted(results)

    # Phase 2: Generate candidate M's for exact verification
    # 1. Top base candidates
    # 2. Scaled versions of top base candidates
    
    m_to_check = set()
    
    # Add top base candidates
    for count, m, k_list in candidates[:200]:
        m_to_check.add(m)
        
    # Add scaled versions
    # Scaling factor d: M_new = M_old * d * sigma(d)
    # We want M_new <= 10^18
    # Try small d's
    scaling_factors = []
    for d in range(2, 50):
        fac_d = factorize(d)
        sig_d = sigma_from_factors(fac_d)
        scaling_factors.append(d * sig_d)
        
    for count, m, k_list in candidates[:50]:
        for sf in scaling_factors:
            m_new = m * sf
            if m_new <= max_m:
                m_to_check.add(m_new)
            else:
                break # Assuming sf is increasing? Not strictly, but small d are small.
                
    # Also try scaling by primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p in small_primes:
        sf = p * (p + 1) # sigma(p) = p+1
        for count, m, k_list in candidates[:50]:
            m_new = m * sf
            if m_new <= max_m:
                m_to_check.add(m_new)
            # Also double scaling
            m_new2 = m * sf * sf
            if m_new2 <= max_m:
                m_to_check.add(m_new2)

    best_count = 0
    best_m = 0
    best_k_list = []
    
    # Phase 3: Exact verification
    # Sort m_to_check by heuristic: prefer those derived from high-count bases?
    # Just iterate. 
    # To save time, we can prioritize M's that are "dense" (many divisors) or from top candidates.
    
    # Let's just iterate over the set. 
    # If the set is too large, we might time out.
    # candidates[:50] * ~100 scalers = 5000 M's.
    # Factorization of M <= 10^18 is fast (~100ms worst case, usually <1ms).
    # Divisor generation is fast.
    # Check is fast.
    # 5000 * 1ms = 5s. This is fine.
    
    for m in m_to_check:
        if time.time() > deadline:
            break
            
        all_k = find_all_solutions_for_m(m)
        
        if len(all_k) > best_count:
            best_count = len(all_k)
            best_m = m
            best_k_list = all_k
            
    if best_count > 0 and best_m > 0:
        valid_k = [k for k in best_k_list if k <= math.isqrt(best_m)]
        valid_k.sort()
        M_out = int(best_m)
        k_out = [int(k) for k in valid_k]
        return {'M': M_out, 'k': k_out}
    
    # Fallback to best base candidate
    if candidates:
        count, m, k_list = candidates[0]
        valid_k = [k for k in k_list if k <= math.isqrt(m)]
        valid_k.sort()
        M_out = int(m)
        k_out = [int(k) for k in valid_k]
        return {'M': M_out, 'k': k_out}
    
    return {'M': 333312, 'k': [336, 372, 434]}

if __name__ == "__main__":
    result = construct()
    print(result)
